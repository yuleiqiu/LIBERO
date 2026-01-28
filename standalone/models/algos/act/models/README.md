# ACT (DETR-VAE) Shape Notes

This note summarizes the tensor shapes and the intent of key steps in the ACT
implementation under this directory. It focuses on the image branch of
`detr_vae.py` and the corresponding logic in `transformer.py`.

## Backbone outputs

In `detr_vae.py`:

- `features, pos = self.backbones[cam_id](image[:, cam_id])`
  - `features`: `List[Tensor]`, each is `(B, C_i, H_i, W_i)`
  - `pos`: `List[Tensor]`, each is `(B, hidden_dim, H_i, W_i)`
- The code then uses the last layer only:
  - `features = features[-1]` -> `(B, backbone.num_channels, H, W)`
  - `pos = pos[-1]` -> `(B, hidden_dim, H, W)`

`input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)` maps
image features into the transformer model dimension.

## Multi-camera fusion (image branch)

For each camera:

- `all_cam_features.append(self.input_proj(features))` -> `(B, hidden_dim, H, W)`
- `all_cam_pos.append(pos)` -> `(B, hidden_dim, H, W)`

Across cameras:

- `src = torch.cat(all_cam_features, axis=3)` -> `(B, hidden_dim, H, W * n_cam)`
- `pos = torch.cat(all_cam_pos, axis=3)` -> `(B, hidden_dim, H, W * n_cam)`

This "tiles" cameras along the width dimension so each camera occupies a
distinct x-range before flattening to tokens.

## How to interpret image tokens (camera tokens)

After `src` is flattened, each token corresponds to a spatial location on the
feature map. You can think of each token as a "patch feature" with a receptive
field in the original image.

With multi-camera inputs, cameras are tiled along width, so:

- Tokens still represent `(h, w)` locations on a feature map.
- The width dimension is expanded to `W * n_cam`.
- Each camera occupies a distinct x-range in this wider map.

The positional embedding provides absolute `(x, y)` coordinates in this wider
map, so the model can infer which camera a token came from based on its x-range.
This is not an explicit camera-ID embedding, but it is a strong positional cue,
along with the visual content itself.

## Extra tokens (latent + proprio)

In `detr_vae.py`:

- `latent_input = self.latent_out_proj(latent_sample)` -> `(B, hidden_dim)`
- `proprio_input = self.input_proj_robot_state(qpos)` -> `(B, hidden_dim)`

In `transformer.py`:

- `addition_input = torch.stack([latent_input, proprio_input], axis=0)`
  -> `(2, B, hidden_dim)`
- Later this is concatenated in front of image tokens on axis=0 (sequence).

## Positional embeddings

`PositionEmbeddingSine` returns `(1, hidden_dim, H, W)` (batch-independent).
So in `transformer.py`:

- `pos_embed = pos_embed.flatten(2)` -> `(1, hidden_dim, H*W)`
- `pos_embed = pos_embed.permute(2, 0, 1)` -> `(H*W, 1, hidden_dim)`
- `pos_embed = pos_embed.repeat(1, bs, 1)` -> `(H*W, B, hidden_dim)`

Then `additional_pos_embed` is added in front:

- `additional_pos_embed` (from `nn.Embedding(2, hidden_dim)`)
  -> `(2, B, hidden_dim)`
- `pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)`
  -> `(H*W + 2, B, hidden_dim)`

## Decoder queries vs paper diagram

The paper figure omits some implementation details that appear in code:

- **Fixed position embeddings** correspond to the 2D sine `pos_embed`. This same
  `pos_embed` is applied in both encoder and decoder (DETR-style), even if the
  figure only annotates it on the decoder side.
- **Query embeddings** are learnable and represent the `chunk_size` output slots
  (action steps). In code this is `query_embed`, passed as `query_pos` to the
  decoder. The diagram may omit it for simplicity.

Separately, the VAE branch uses a 1D sinusoidal `pos_table` for the action/robot
state sequence; this is distinct from the 2D image `pos_embed`.

## Transformer input/output shapes

After flattening image features:

- `src` becomes `(H*W, B, hidden_dim)`
- with extra tokens prepended: `(H*W + 2, B, hidden_dim)`

Encoder:

- `memory = self.encoder(src, pos=pos_embed)`
  -> same shape as `src`, but each token is context-mixed.
  "Global context" means every token can attend to every other token
  (no mask), so each output token contains information from the whole sequence.

Decoder:

- `query_embed.weight`: `(num_queries, hidden_dim)` where `num_queries = chunk_size`
- `tgt = zeros_like(query_embed)` -> `(num_queries, B, hidden_dim)`
- `hs` (with `return_intermediate_dec=True`):
  - before transpose: `(num_layers, num_queries, B, hidden_dim)`
  - after `hs.transpose(1, 2)`: `(num_layers, B, num_queries, hidden_dim)`

## Quick reference

- `src`: `(H*W + 2, B, hidden_dim)`
- `pos_embed`: `(H*W + 2, B, hidden_dim)`
- `query_embed`: `(num_queries, B, hidden_dim)`
- `memory`: `(H*W + 2, B, hidden_dim)`
- `hs`: `(num_layers, B, num_queries, hidden_dim)`

## Decoder attention flow (tgt/query/memory)

The decoder works in two attention stages per layer:

1) **Self-attention among queries**
   - `tgt` starts as all zeros with shape `(num_queries, B, hidden_dim)`.
   - `query_pos` (learnable `query_embed`) gives each query its identity.
   - Self-attention lets queries interact and coordinate across time steps.

2) **Cross-attention from queries to memory**
   - `query = tgt + query_pos`
   - `key = memory + pos` (pos is fixed 2D positional embedding for image tokens)
   - `value = memory`
   - This lets each action query read the global, contextualized encoder tokens
     (latent + proprio + image patches).

Finally a feed-forward layer refines each query token. The resulting per-query
features are mapped to actions by the `action_head`.
