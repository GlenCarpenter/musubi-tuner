# Krea 2 Edit Full-Checkpoint Training Rollout Plan

## Goal

Add full-checkpoint training for the `krea2_edit` architecture now that its LoRA workflow and `comfyui-krea2edit` inference compatibility are verified.

The rollout must:

- preserve existing Krea 2 Text-to-Image LoRA and full-checkpoint behavior;
- preserve the verified Krea 2 Edit dataset, cache, geometry, and conditioning contracts;
- optimize the Krea 2 RAW DiT directly while keeping the VAE and Qwen3-VL text encoder frozen;
- save a complete Krea 2 DiT checkpoint with the same tensor-key contract as the input RAW checkpoint;
- keep LoRA and full-checkpoint commands unambiguous and independently testable.

The verified LoRA implementation in [krea2-edit-lora-plan.md](./krea2-edit-lora-plan.md) is the behavioral reference. Full training changes which parameters are optimized and how checkpoints are saved; it must not change how edit examples are conditioned or supervised.

The original LoRA rollout established an important evidence boundary: short training runs prove integration, gradient flow, serialization, and loader compatibility, but not visual convergence. Visual acceptance for full training requires appropriately converged runs rather than a fixed small step count.

## Initial Support Boundary

The first supported path is deliberately narrow:

- Krea 2 RAW DiT only;
- BF16 mixed-precision training;
- SDPA attention;
- gradient checkpointing;
- dataset batch size `1`, with gradient accumulation for larger effective batches;
- one or two ordered references;
- precomputed edit latent and fixed-scale grounded-text caches;
- epoch, step, and final full-checkpoint saves;
- training-state save and resume.

The following remain disabled until separately validated:

- Turbo checkpoint training;
- scaled FP8 and ConvRot INT8, because quantized base weights cannot be treated as ordinary trainable full-precision parameters;
- edit-aware sample generation;
- flip augmentation;
- dataset batch sizes above `1`;
- any attention backend, block-swap mode, or `torch.compile` combination not covered by a full-training backward and save/resume test.

Unsupported combinations must fail during argument validation rather than silently falling back.

## Architecture

### Entrypoints and ownership

Add a dedicated full-training entrypoint:

```text
krea2_edit_train.py
src/musubi_tuner/krea2_edit_train.py
```

The full trainer should specialize the regular Krea 2 full-checkpoint trainer and reuse the edit-specific behavior currently owned by `Krea2EditNetworkTrainer`:

- `krea2_edit` architecture selection;
- dataset and cache preflight;
- ordered reference extraction;
- edit image-token packing and positional construction;
- grounded Qwen3-VL context loading;
- target-output slicing;
- target-only flow-matching loss.

Extract those behaviors into narrow shared hooks or a mixin if necessary so the LoRA and full trainers execute the same conditioning path. Do not duplicate the packing and loss-routing implementation in two trainers.

### Trainable state

Full training must make the DiT parameters trainable before optimizer construction and pass those parameters directly to the optimizer. The VAE and text encoder remain frozen and are used only by the existing cache commands or optional future previews.

Preflight must reject LoRA-only arguments, including:

- `--network_module`;
- `--network_weights`;
- `--network_dim` and `--network_alpha`;
- `--network_dropout` and `--network_args`;
- `--dim_from_weights`;
- `--base_weights` and `--base_weights_multiplier`.

A startup summary must report the number of trainable tensors and parameters. A focused test must prove that representative DiT parameters receive gradients while reference inputs and cached conditioning remain unsupervised inputs.

### Conditioning and loss parity

For a fixed batch, timestep, and noise seed, the full trainer must build the same model inputs and loss target as the verified LoRA trainer:

```text
[clean_ref_1 | clean_ref_2 | noisy_target | grounded_text | padding]
```

Only the target latent is noised. Reference latents remain clean. Predictions for reference tokens are discarded, and only the unpacked target prediction contributes to the existing flow-matching loss against `noise - clean_target`.

Parity tests should compare packed tokens, masks, positions, target slices, loss targets, and scalar loss between the LoRA and full-training paths before optimizer-specific behavior is introduced.

The geometry baseline is the corrected `comfyui-krea2edit` v1.2.5 fit protocol: crop before resize, preserve fractional centering offsets, and keep one token per `16x16` pixels. Musubi training uses bilinear antialiased interpolation while ComfyUI inference uses bicubic antialiased interpolation; this documented kernel difference is intentional and must not be mistaken for a geometry mismatch.

### Dataset configuration authority

The generated dataset TOML is authoritative for `reference_directories`. GUI bookkeeping fields persisted in a runtime TOML must not flow through argparse fallback and populate singular `reference_directory` at the same time.

Preserve the existing Krea 2 Edit blueprint rule that excludes both singular and plural reference-directory fields from argparse fallback. Add a regression covering a runtime TOML with singular GUI bookkeeping plus a dataset TOML with plural ordered references. Do not fix collisions by removing GUI persistence, rewriting user runtime TOMLs, or changing fallback behavior for other architectures.

### Checkpoint contract

Each model save must contain the complete DiT state, not adapter tensors. Saving must:

- unwrap Accelerate and compiled wrappers without changing key names;
- preserve the standard Krea 2 RAW tensor-key schema and expected dtype;
- avoid LoRA prefixes or merged-adapter artifacts;
- reuse the base Krea 2 ModelSpec architecture and implementation identity rather than inventing a distinct loader-visible Image Edit model identity;
- record the `krea2_edit` training task only in task-specific metadata that does not affect base-model detection;
- support memory-efficient safetensors output;
- retain existing checkpoint retention, Hugging Face upload, and asynchronous upload behavior from regular Krea 2 full training;
- save optimizer, scheduler, scaler, RNG, epoch, and global-step state through the existing resume mechanism.

Checkpoint validation must compare input and output key sets, load the saved model through the standard Krea 2 loader, run an edit forward pass, and resume training for at least one optimizer step.

## Rollout Phases

### Phase 1: Establish the full-training baseline

- Identify the regular Krea 2 full trainer revision that will be the parent implementation.
- Record its supported precision, attention, block-swap, compile, save, and resume modes.
- Add regression tests proving its Text-to-Image command and checkpoint schema are unchanged before introducing Image Edit.
- Capture one deterministic Krea 2 Edit LoRA batch as the conditioning and loss-parity fixture.
- Preserve the existing ModelSpec mapping from both `krea2` and `krea2_edit` to the base Krea 2 architecture and implementation.
- Preserve the dataset-blueprint exclusion that keeps runtime GUI reference fields out of Krea 2 Edit dataset fallback.
- Restore or recreate the durable Phase 7 identity, outpaint, and ordered two-reference harness plus focused ComfyUI compatibility tests; keep generated data, caches, checkpoints, images, and metrics ignored.
- Retain the existing compatibility validator's checks for complete LoRA triplets, unambiguous aliases, rank/base shape compatibility, scalar alpha, and raw or `diffusion_model.` base namespaces as LoRA regression coverage.

**Exit criterion:** the parent full trainer and the verified edit forward path each have a focused baseline test.

### Phase 2: Share the edit training path

- Extract edit cache preflight, reference handling, and `call_dit` behavior into reusable hooks or a mixin.
- Keep `Krea2EditNetworkTrainer` behavior and CLI validation unchanged.
- Add the full trainer using the regular Krea 2 full-training loop plus the shared edit hooks.
- Add the top-level wrapper and parser wiring.

**Exit criterion:** LoRA tests remain green, and both trainers produce identical model inputs and loss for the parity fixture.

### Phase 3: Direct DiT optimization

- Enable gradients on all intended Krea 2 DiT parameters.
- Build the optimizer from DiT parameters instead of a network module.
- Keep the VAE, cached text outputs, and reference latent tensors outside the optimizer.
- Reject LoRA, quantization, preview, augmentation, and unsupported backend arguments during preflight.
- Verify gradient accumulation, clipping, scheduler stepping, and distributed synchronization use the regular full-training implementation.

**Exit criterion:** a tiny CPU model completes forward, backward, optimizer step, and zero-grad with gradients confined to the DiT.

### Phase 4: Full-checkpoint save and resume

- Save step, epoch, and final checkpoints using the regular Krea 2 full-checkpoint serializer.
- Normalize compiled-wrapper keys before serialization.
- Verify metadata distinguishes Image Edit training while preserving standard Krea 2 model loading.
- Exercise checkpoint retention and training-state save/resume.
- Add failure tests for partial, adapter-only, or key-renamed output.

**Exit criterion:** a saved checkpoint has complete key parity, reloads through the standard Krea 2 loader, and resumes with matching global step and optimizer state.

### Phase 5: Real-model BF16 validation

- Run one-step and short overfit tests with one reference.
- Repeat with two references and a non-square target that produces fractional centering offsets.
- Cover identity, outpaint, and semantic edit fixtures used to verify the LoRA workflow.
- Compare initial full-trainer loss against the LoRA trainer with adapters disabled under the same seed.
- Confirm loss decreases during a short overfit and saved checkpoints produce finite outputs in the target inference nodes.
- Measure peak VRAM, host RAM, checkpoint size, and save time.

The one-step and short-overfit runs in this phase are engineering smoke tests. They are not evidence of visual parity or sufficient convergence.

**Exit criterion:** the BF16 full trainer overfits the compatibility fixtures, saves reloadable checkpoints, and preserves expected reference alignment in `comfyui-krea2edit`.

### Phase 6: Memory and execution features

Validate features independently, in this order:

1. gradient checkpointing;
2. BF16 block swap with backward support;
3. `torch.compile`;
4. additional attention backends supported by regular Krea 2 full training.

For each mode, require a real optimizer step, checkpoint save, clean reload without that mode enabled, and a finite edit inference result. Keep scaled FP8 and ConvRot INT8 rejected unless a future optimizer and serialization design explicitly supports updating quantized base weights.

**Exit criterion:** every advertised mode has an isolated real-model test and emits its active path in stdout.

### Phase 7: CLI documentation and GUI rollout

- Add a full-checkpoint command to the Krea 2 Edit CLI documentation.
- Explain output format, expected storage, optimizer-state storage, supported precision, and resume behavior.
- Add a LoRA / Full Checkpoint selector for Krea 2 Image Edit without changing the Text-to-Image selector behavior.
- Keep task mode (Text-to-Image / Image Edit) independent from training mode (LoRA / Full Checkpoint); do not infer either setting from the other during load or callbacks.
- Select `krea2_edit_train.py` only for Image Edit plus Full Checkpoint.
- Hide or disable LoRA-only controls and unsupported full-training controls.
- Preserve the selected mode through hand-saved and runtime TOML round trips.
- Preserve GUI reference-directory bookkeeping for round trips while keeping those fields outside backend dataset fallback.
- Avoid parent-event and chained-callback validation that can compare stale task/training values across separate Gradio server round trips; validate the final combined state at the command boundary.
- Keep the mode behind an experimental label until Phase 8 passes.

**Exit criterion:** printed commands, GUI construction, and TOML round-trip tests select the correct entrypoint and never leak LoRA arguments into full training.

### Phase 8: End-to-end release gate

- Train fresh one-reference and two-reference checkpoints from the Krea 2 RAW base.
- Load each checkpoint with the regular Krea 2 loader and `comfyui-krea2edit` v1.2.4 or newer.
- Compare edit geometry and qualitative behavior with the verified LoRA baseline.
- Use properly converged runs for visual acceptance; do not promote one-step or short-overfit observations into parity claims.
- Resume a stopped run and verify continuity of step count and loss.
- Confirm existing Krea 2 Text-to-Image LoRA, Text-to-Image full training, and Image Edit LoRA smoke tests remain green.
- Publish measured hardware requirements and known unsupported combinations.

**Exit criterion:** full checkpoints load without conversion, preserve the verified fit protocol, resume correctly, and introduce no regression in the three existing Krea 2 training modes.

## Testing Matrix

| Layer | Required coverage |
| --- | --- |
| Unit | Argument guards, trainable parameter selection, reference ordering, target-only slicing, ModelSpec identity, key normalization, dataset fallback isolation |
| Parity | LoRA versus full packed tokens, masks, positions, targets, and pre-update loss |
| Tiny model | One/two-reference forward, backward, optimizer step, gradient accumulation, save, load, resume |
| Real model | BF16 one-step, short overfit, one/two references, fractional offsets, full checkpoint reload; smoke status only until convergence |
| Compatibility | Standard Krea 2 loader, base Krea 2 ModelSpec detection, and `comfyui-krea2edit` inference |
| Regression | Text-to-Image LoRA, Text-to-Image full checkpoint, Image Edit LoRA, GUI command generation, runtime/dataset TOML reference isolation |

## Release Criteria

Full-checkpoint Image Edit training is ready to leave experimental status only when:

- conditioning and loss parity with the verified LoRA path are automated;
- every advertised mode completes backward, optimizer step, save, reload, and resume;
- output checkpoints retain the standard Krea 2 RAW key contract;
- one-reference and two-reference checkpoints work in the target inference nodes without conversion;
- Text-to-Image LoRA/full and Image Edit LoRA regression suites remain green;
- documentation lists measured resource requirements and all rejected combinations.

## Attribution

The conditioning contract remains based on `lbouaraba/krea2edit-trainer` and its geometry documentation. That project is Apache-2.0 and includes code adapted from the Apache-2.0 Krea 2 reference implementation. Any directly adapted source must retain the required license and NOTICE attribution.