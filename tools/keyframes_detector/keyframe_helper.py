# From https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py

# Standard library imports
import json
import os
import argparse
import pprint

# Third-party imports
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored
import h5py
import imageio

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# from rlbench.demo import Demo
from typing import Dict, List, Optional, Tuple

def print_h5_structure(group, indent=0):
    """Recursively prints the structure of an HDF5 group."""
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{'  ' * indent}- {key} (Group)")
            print_h5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{'  ' * (indent)}- {key} (Dataset)")
            print(f"{'  ' * (indent + 1)}  Shape: {item.shape}")
            print(f"{'  ' * (indent + 1)}  Dtype: {item.dtype}")
            if item.dtype.names:
                print(f"{'  ' * (indent + 1)}  Fields: {item.dtype.names}")

def _keypoint_discovery(
    demo,
    motion_delta=0.02,
    gripper_delta=5e-3,
    min_stationary_frames=4,
    min_motion_frames=2,
    min_keypoint_spacing=5,
) -> List[int]:
    """Return transition frames by segmenting still/motion phases and gripper toggles."""

    joint_states = np.asarray(demo['obs']['joint_states'][()])
    gripper_command = np.asarray(demo['actions'][()])[:, -1]

    episode_keypoints: List[int] = [0]
    phase = "moving"
    still_count = 0
    move_count = 0
    last_keypoint = 0
    last_gripper_event = gripper_command[0]

    for i in range(1, len(joint_states)):
        velocity = np.linalg.norm(joint_states[i] - joint_states[i - 1])
        is_still = velocity < motion_delta

        if is_still:
            still_count += 1
            move_count = 0
            if phase != "still" and still_count >= min_stationary_frames:
                phase = "still"
        else:
            move_count += 1
            still_count = 0
            if phase == "still" and move_count == 1 and i - last_keypoint >= min_keypoint_spacing:
                episode_keypoints.append(i)
                last_keypoint = i
            if phase != "moving" and move_count >= min_motion_frames:
                phase = "moving"

        if abs(gripper_command[i] - last_gripper_event) >= gripper_delta and i - last_keypoint >= min_keypoint_spacing:
            episode_keypoints.append(i)
            last_keypoint = i
            last_gripper_event = gripper_command[i]

    if episode_keypoints[-1] != len(joint_states) - 1:
        episode_keypoints.append(len(joint_states) - 1)

    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def _get_ee_positions(demo) -> np.ndarray:
    """Extract end-effector positions from the demo, regardless of naming."""
    obs_group = demo["obs"]
    if "ee_pos" in obs_group:
        return np.asarray(obs_group["ee_pos"][()])
    if "ee_states" in obs_group:
        return np.asarray(obs_group["ee_states"][()])[:, :3]
    raise KeyError("Demo does not contain 'ee_pos' or 'ee_states' observations.")

def _find_threshold_crossing(
    signal: np.ndarray,
    threshold: float,
    comparison: str,
    start_index: int = 0,
) -> Optional[int]:
    """Return the first index >= start_index where signal satisfies the comparison."""
    if start_index >= len(signal):
        return None
    portion = signal[start_index:]
    if comparison == "gte":
        candidates = np.where(portion >= threshold)[0]
    elif comparison == "lte":
        candidates = np.where(portion <= threshold)[0]
    else:
        raise ValueError(f"Unsupported comparison {comparison}")
    if candidates.size == 0:
        return None
    return start_index + int(candidates[0])

def _enforce_monotonic_phases(
    raw_points: List[int],
    total_frames: int,
    min_gap: int,
) -> List[int]:
    """Clamp candidate points so that phases are non-overlapping and ordered."""
    if total_frames == 0:
        return [0] * len(raw_points)

    ordered = [raw_points[0]]
    n_points = len(raw_points)
    for idx in range(1, n_points - 1):
        remaining_boundaries = (n_points - 1) - idx
        min_allowed = ordered[-1] + min_gap
        max_allowed = (total_frames - 1) - remaining_boundaries * min_gap
        if min_allowed > max_allowed:
            candidate = max_allowed
        else:
            candidate = max(min_allowed, min(raw_points[idx], max_allowed))
        ordered.append(int(candidate))

    ordered.append(total_frames - 1)
    return ordered

def _segment_grasp_phases(
    demo,
    closing_threshold: float = 0.5,
    opening_threshold: float = -0.5,
    lift_height_delta: float = 0.05,
    min_phase_frames: int = 5,
    max_grasp_search: int = 150,
) -> Tuple[List[int], Dict[str, Tuple[int, int]]]:
    """Split a trajectory into approach, grasp, lift, and place phases."""
    actions = np.asarray(demo["actions"][()])
    ee_pos = _get_ee_positions(demo)
    timesteps = actions.shape[0]
    gripper_signal = actions[:, -1]
    if timesteps == 0:
        raise ValueError("Demo contains zero timesteps.")

    closing_idx = _find_threshold_crossing(
        gripper_signal, closing_threshold, "gte", 0
    )
    if closing_idx is None:
        closing_idx = max(1, int(0.25 * (timesteps - 1)))

    grasp_search_stop = min(timesteps, closing_idx + max_grasp_search)
    if grasp_search_stop <= closing_idx:
        contact_idx = closing_idx
    else:
        rel_argmin = int(np.argmin(ee_pos[closing_idx:grasp_search_stop, 2]))
        contact_idx = closing_idx + rel_argmin

    lift_idx = contact_idx
    baseline_height = ee_pos[contact_idx, 2]
    for idx in range(contact_idx, timesteps):
        if ee_pos[idx, 2] - baseline_height >= lift_height_delta:
            lift_idx = idx
            break
    else:
        lift_idx = min(timesteps - 2, contact_idx + min_phase_frames)

    place_start_idx = _find_threshold_crossing(
        gripper_signal, opening_threshold, "lte", lift_idx
    )
    if place_start_idx is None:
        place_start_idx = max(
            lift_idx + min_phase_frames, int(0.85 * (timesteps - 1))
        )

    raw_points = [0, closing_idx, lift_idx, place_start_idx, timesteps - 1]
    ordered_points = _enforce_monotonic_phases(
        raw_points, timesteps, min_phase_frames
    )

    phase_names = ["approach", "grasp", "lift", "place"]
    segments = {
        phase: (ordered_points[i], ordered_points[i + 1])
        for i, phase in enumerate(phase_names)
    }
    return ordered_points, segments

PHASE_NAME_ORDER = ["approach", "grasp", "lift", "place"]
PHASE_STYLES = {
    "approach": ("Approach", (52, 152, 219)),
    "grasp": ("Grasp", (231, 76, 60)),
    "lift": ("Lift", (46, 204, 113)),
    "place": ("Place", (155, 89, 182)),
    "done": ("Done", (149, 165, 166)),
}

def _phase_display(phase: str) -> Tuple[str, Tuple[int, int, int]]:
    """Return a user-friendly label and color for a phase."""
    label, color = PHASE_STYLES.get(phase, (phase.title(), (255, 255, 255)))
    return label, color

def _phase_for_frame(frame_idx: int, phase_points: List[int], phase_names: List[str]) -> str:
    """Determine the active phase for a frame index."""
    last_idx = len(phase_names) - 1
    for idx, phase in enumerate(phase_names):
        start = phase_points[idx]
        end = phase_points[idx + 1]
        is_last = idx == last_idx
        if is_last:
            if start <= frame_idx <= end:
                return phase
        else:
            if start <= frame_idx < end:
                return phase
    return phase_names[-1]

def _compose_camera_frame(agentview_img: np.ndarray, eye_in_hand_img: Optional[np.ndarray]) -> np.ndarray:
    """Create a composite frame (agentview | eye-in-hand) and flip vertically for display."""
    agent_frame = np.asarray(agentview_img)[::-1]
    if eye_in_hand_img is None:
        return agent_frame
    eye_frame = np.asarray(eye_in_hand_img)[::-1]
    return np.hstack((agent_frame, eye_frame))

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Get width and height for a text string."""
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    width, height = draw.textsize(text, font=font)
    return width, height

def _annotate_frame(
    frame: np.ndarray,
    main_text: str,
    sub_text: Optional[str],
    text_color: Tuple[int, int, int],
    box_color: Tuple[int, int, int] = (0, 0, 0),
    position: str = "bottom",
) -> np.ndarray:
    """Attach a single-line banner above or below the frame."""
    frame_img = Image.fromarray(frame)
    frame_width, frame_height = frame_img.size
    font = ImageFont.load_default()
    padding = 8
    banner_text = main_text if not sub_text else f"{main_text} | {sub_text}"
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_width, text_height = _measure_text(dummy_draw, banner_text, font)
    banner_height = text_height + 2 * padding
    banner_img = Image.new("RGB", (frame_width, banner_height), color=box_color)
    banner_draw = ImageDraw.Draw(banner_img)
    text_x = padding
    text_y = padding
    banner_draw.text((text_x, text_y), banner_text, fill=text_color, font=font)

    combined = Image.new("RGB", (frame_width, frame_height + banner_height))
    if position == "top":
        combined.paste(banner_img, (0, 0))
        combined.paste(frame_img, (0, banner_height))
    else:
        combined.paste(frame_img, (0, 0))
        combined.paste(banner_img, (0, frame_height))
    return np.asarray(combined)

def _render_phase_video(
    demo,
    phase_points: List[int],
    output_path: str,
    fps: int = 30,
    pause_seconds: float = 1.0,
    phase_names: Optional[List[str]] = None,
    label_position: str = "bottom",
) -> None:
    """Create a video that overlays the current phase and pauses at transitions."""
    phase_names = phase_names or PHASE_NAME_ORDER
    obs_group = demo["obs"]
    if "agentview_rgb" not in obs_group:
        print("[Warning] agentview_rgb observations not found; skipping video export.")
        return

    agentview_imgs = np.asarray(obs_group["agentview_rgb"][()])
    eye_imgs = np.asarray(obs_group["eye_in_hand_rgb"][()]) if "eye_in_hand_rgb" in obs_group else None
    total_frames = agentview_imgs.shape[0]
    if total_frames == 0:
        print("[Warning] No RGB frames available; skipping video export.")
        return

    max_frame_idx = total_frames - 1
    video_phase_points = [min(p, max_frame_idx) for p in phase_points]

    pause_frames = int(round(max(0.0, pause_seconds) * fps))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)

    transitions = {}
    extended_names = phase_names + ["done"]
    for idx in range(1, len(video_phase_points)):
        prev_phase = extended_names[idx - 1]
        next_phase = extended_names[idx] if idx < len(extended_names) else "done"
        transitions[video_phase_points[idx]] = (prev_phase, next_phase)

    for frame_idx in range(total_frames):
        agent_frame = agentview_imgs[frame_idx]
        eye_frame = eye_imgs[frame_idx] if eye_imgs is not None else None
        composite = _compose_camera_frame(agent_frame, eye_frame)
        phase = _phase_for_frame(frame_idx, video_phase_points, phase_names)
        label, color = _phase_display(phase)
        annotated = _annotate_frame(
            composite,
            f"Phase: {label}",
            f"Frame {frame_idx}",
            text_color=color,
            position=label_position,
        )
        writer.append_data(annotated)

        if pause_frames > 0 and frame_idx in transitions:
            prev_phase, next_phase = transitions[frame_idx]
            next_label, next_color = _phase_display(next_phase)
            transition_label = f"Transition: {_phase_display(prev_phase)[0]} -> {next_label}"
            pause_frame = _annotate_frame(
                composite,
                transition_label,
                # f"Holding for {pause_seconds:.1f}s",
                f"Frame {frame_idx}",
                text_color=next_color,
                box_color=(30, 30, 30),
                position=label_position,
            )
            for _ in range(pause_frames):
                writer.append_data(pause_frame)

    writer.close()
    print(f"[Video] Saved phase visualization to {output_path}")

def main():
    pp = pprint.PrettyPrinter(indent=2)

    # Allow specifying task_id via command line argument
    parser = argparse.ArgumentParser(description='Visualize keyframe of a demonstration')
    parser.add_argument('--task-id', type=int, default=0, help='Task ID to visualize')
    parser.add_argument('--benchmark', type=str, default="libero_object", help='Benchmark name')
    parser.add_argument('--output-dir', type=str, default="tools/keyframes_detector/tmp", help='Output directory for videos')
    parser.add_argument('--motion-delta', type=float, default=0.02, help='Max joint delta that still counts as stationary')
    parser.add_argument('--gripper-delta', type=float, default=5e-3, help='Gripper command delta required to log a keypoint')
    parser.add_argument('--min-stationary-frames', type=int, default=4, help='Frames of stillness before a new motion phase may start')
    parser.add_argument('--min-motion-frames', type=int, default=2, help='Frames of sustained motion to exit the still phase')
    parser.add_argument('--min-keypoint-spacing', type=int, default=5, help='Minimum frame gap between consecutive keypoints')
    parser.add_argument('--closing-threshold', type=float, default=0.5, help='Min gripper command value that counts as closing')
    parser.add_argument('--opening-threshold', type=float, default=-0.5, help='Max gripper command value that counts as opening')
    parser.add_argument('--lift-height-delta', type=float, default=0.05, help='Meters of upward motion required to deem the object lifted')
    parser.add_argument('--min-phase-frames', type=int, default=8, help='Minimum number of frames for each macro phase')
    parser.add_argument('--max-grasp-search', type=int, default=150, help='Frames to search after closing for the grasp contact point')
    parser.add_argument('--video-fps', type=int, default=30, help='FPS for the rendered phase video')
    parser.add_argument('--transition-pause', type=float, default=1.0, help='Seconds to pause at each phase transition')
    parser.add_argument('--label-position', type=str, choices=['top', 'bottom'], default='bottom', help='Where to place the phase banner relative to the video frames')
    parser.add_argument('--skip-video', action='store_true', help='Skip generating the annotated video')
    args = parser.parse_args()

    datasets_default_path = get_libero_path("datasets")
    bddl_dir = get_libero_path("bddl_files")
    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark]()
    task_id = args.task_id
    task = benchmark_instance.get_task(task_id)

    demo_files = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(task_id))
    demo_h5 = h5py.File(demo_files, "r")["data"]
    demo_keys = [k for k in demo_h5.keys() if k.startswith("demo_")]
    num_demos = len(demo_keys)
    print(f"Number of demos for task {task_id}: {num_demos}")

    bddl_file_name = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
    print(f"Using BDDL file: {bddl_file_name}")

    demo = demo_h5[demo_keys[0]]
    # gripper_state = demo["actions"][:, -1]
    # for state in gripper_state:
    #     print(state)

    # Detect keypoints in the demonstration
    print(f"\nDetecting keypoints for {demo_keys[0]}...")
    keypoints = _keypoint_discovery(
        demo,
        motion_delta=args.motion_delta,
        gripper_delta=args.gripper_delta,
        min_stationary_frames=args.min_stationary_frames,
        min_motion_frames=args.min_motion_frames,
        min_keypoint_spacing=args.min_keypoint_spacing,
    )

    phase_points, phase_segments = _segment_grasp_phases(
        demo,
        closing_threshold=args.closing_threshold,
        opening_threshold=args.opening_threshold,
        lift_height_delta=args.lift_height_delta,
        min_phase_frames=args.min_phase_frames,
        max_grasp_search=args.max_grasp_search,
    )
    
    # Print some statistics
    print(f"\nDemo statistics:")
    print(f"  Total timesteps: {len(demo['obs']['gripper_states'])}")
    print(f"  Number of keypoints: {len(keypoints)}")
    print(f"  Keypoint indices: {keypoints}")
    print(f"\nPhase split keypoints (approach/grasp/lift/place/done): {phase_points}")
    for phase, (start_idx, end_idx) in phase_segments.items():
        print(f"  {phase:>8s}: frames {start_idx} -> {end_idx}")

    if not args.skip_video:
        os.makedirs(args.output_dir, exist_ok=True)
        video_filename = f"{task.problem_folder}_task{task_id}_{demo_keys[0]}_phases.mp4"
        video_path = os.path.join(args.output_dir, video_filename)
        _render_phase_video(
            demo,
            phase_points,
            video_path,
            fps=args.video_fps,
            pause_seconds=args.transition_pause,
            phase_names=PHASE_NAME_ORDER,
            label_position=args.label_position,
        )
    
    # # Analyze gripper states
    # gripper_states = demo['obs']['gripper_states'][:, 0]
    # print(f"  Gripper state range: [{gripper_states.min():.8f}, {gripper_states.max():.8f}]")
    # print(f"  Unique gripper states: {np.unique(gripper_states)}")


if __name__ == "__main__":
    main()
