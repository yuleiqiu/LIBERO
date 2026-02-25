# Dataset Format

LIBERO datasets are stored as HDF5 files (`*_demo.hdf5`). Each file contains a `data/` group
with multiple demo keys (e.g. `demo_0`, `demo_1`, …).

## Per-demo structure

Inspected from `libero_object_single/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`
(T = number of timesteps, varies per demo).

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `actions` | (T, 7) | float64 | End-effector delta actions: 3 pos + 3 ori + 1 gripper |
| `dones` | (T,) | uint8 | Episode termination flags |
| `rewards` | (T,) | uint8 | Per-step reward signal |
| `robot_states` | (T, 9) | float64 | Proprioceptive robot state (ee_pos 3 + ee_ori 3 + gripper 2 + ?) |
| `states` | (T, 45) | float64 | Full simulator state vector |
| `obs/agentview_rgb` | (T, 128, 128, 3) | uint8 | Third-person camera image |
| `obs/eye_in_hand_rgb` | (T, 128, 128, 3) | uint8 | Wrist camera image |
| `obs/ee_pos` | (T, 3) | float64 | End-effector position |
| `obs/ee_ori` | (T, 3) | float64 | End-effector orientation (Euler) |
| `obs/ee_states` | (T, 6) | float64 | Combined ee_pos + ee_ori |
| `obs/gripper_states` | (T, 2) | float64 | Gripper finger positions |
| `obs/joint_states` | (T, 7) | float64 | Joint angles |

## Utilities

- `standalone/inspect/print_obs_keys.py` — print obs keys and shapes for any demo file
- `standalone/inspect/print_demo_actions.py` — inspect action sequences
