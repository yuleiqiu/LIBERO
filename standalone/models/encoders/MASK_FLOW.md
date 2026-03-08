# DP Mask Flow

This note documents how segmentation masks flow through the current Diffusion Policy
encoder stack, and where to debug when mask-conditioned training does not behave as
expected.

## Current Mask Semantics

Source semantic values from `get_segmentation_of_interest(...)`:

- `1`: object of interest
- `0`: other mapped instances
- `-1`: background / unmapped pixels

Training-time encoder semantic values after preprocessing in `ObsEncoder._get_mask(...)`:

- `1`: object of interest
- `0`: irrelevant region

In other words, training currently uses a binary object-of-interest mask.

## End-to-End Call Chain

### 1. Dataset writes mask keys into HDF5

Example producer:

- `scripts/create_dataset_with_segmentation.py`

Expected HDF5 keys under each `demo_x/obs/`:

- `agentview_rgb`
- `eye_in_hand_rgb`
- `agentview_segmentation_of_interest`
- `eye_in_hand_segmentation_of_interest`

### 2. Train config names image keys and mask keys

Relevant config fields:

- `data.image_keys`
- `data.mask_keys`

`mask_keys` is parallel to `image_keys`.
Empty string means that image key has no paired mask.

### 3. `standalone/train.py` loads mask keys into the dataset

`train.py`:

- parses `cfg.data.mask_keys`
- appends active mask keys into `all_keys`
- builds `HDF5SequenceDataset(obs_keys=all_keys, ...)`
- passes `mask_keys` into `build_policy(...)`

This means masks are loaded as ordinary observation entries inside `batch["obs"]`.

### 4. `DiffusionPolicy` passes mask keys into the obs encoder

`standalone/models/policy/policy_factory.py`
-> `DiffusionPolicy(..., mask_keys=mask_keys)`

`standalone/models/policy/diffusion_policy.py`
-> `build_obs_encoder(..., mask_keys=mask_keys)`

### 5. `build_obs_encoder(...)` creates `image_key -> mask_key` mapping

`standalone/models/encoders/obs.py`

The mapping is built from the parallel `image_keys` and `mask_keys` lists:

- `agentview_rgb -> agentview_segmentation_of_interest`
- `eye_in_hand_rgb -> eye_in_hand_segmentation_of_interest`

### 6. `ObsEncoder._get_mask(...)` standardizes mask format

`standalone/models/encoders/obs.py`

For each image key:

- fetch paired mask from `obs[mask_key]`
- convert `(B, H, W)` to `(B, 1, H, W)` if needed
- binarize with `m = (m > 0).float()`

This is the canonical preprocessing step for mask-conditioned training.

### 7. `ObsEncoder.forward(...)` passes mask only to `DPImageEncoder`

`standalone/models/encoders/obs.py`

Image path:

- read RGB tensor
- read paired binary mask
- call `encoder(x, mask=mask)` if encoder is `DPImageEncoder`
- otherwise ignore mask

Important:

- mask is only used when the image encoder type is `dp_resnet`
- plain `ImageEncoder` does not consume mask

### 8. `DPImageEncoder.forward(...)` applies the mask on feature maps

`standalone/models/encoders/image.py`

Processing order:

1. Optionally random-crop RGB
2. If a mask is present, apply the exact same crop
3. Run ResNet backbone on RGB
4. Downsample mask to feature-map resolution
5. Reweight feature map:

```python
h = h * (mask_alpha + (1.0 - mask_alpha) * mask_low)
```

Effect:

- object region (`mask=1`) keeps full weight
- irrelevant region (`mask=0`) is suppressed to `mask_alpha`

Default behavior is soft suppression, not hard removal.

### 9. Masked visual features become DP global conditioning

`standalone/models/algos/dp/core/diffusion_model.py`

Flow:

- `obs_encoder(this_nobs)`
- flatten encoded observation history
- build `global_cond`
- feed `global_cond` into `ConditionalUnet1D`

The mask does not define a separate loss.
It only changes the visual condition features used by the diffusion model.

## Runtime Debug Checklist

If mask-conditioned training seems inactive, check these in order:

1. HDF5 contains `*_segmentation_of_interest` under each `demo_x/obs/`.
2. `cfg.data.mask_keys` matches `cfg.data.image_keys` by position.
3. `train.py` debug print for `batch["obs"].keys()` includes mask keys.
4. Policy image encoder type is `dp_resnet`.
5. `ObsEncoder._get_mask(...)` returns binary values only.
6. `DPImageEncoder.forward(...)` does not raise the binary-mask assertion.
7. `mask_alpha` is not so high that suppression becomes negligible.

## Current Limitation

The main env rollout path in `standalone/rollout_env.py` now supports mask-conditioned
DP encoding and generates binary object-of-interest masks online from instance
segmentations.

Other utility entrypoints, such as `standalone/rollout.py`, are still separate and do
not represent the full env-interaction path.
