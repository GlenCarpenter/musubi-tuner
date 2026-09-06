# Krea 2 Edit CLI Training

The three Krea 2 Edit scripts provide the complete CLI workflow without requiring the GUI:

1. Cache target and reference-image latents.
2. Cache image-grounded Qwen3-VL outputs.
3. Train the LoRA.

## Dataset Layout

Create the following directory structure:

```text
D:/datasets/krea2-edit/
├── targets/
│   ├── 0001.png
│   ├── 0001.txt
│   ├── 0002.png
│   └── 0002.txt
├── sources/
│   ├── 0001.png
│   └── 0002.png
├── sources_b/          # Optional second reference
│   ├── 0001.png
│   └── 0002.png
└── cache/
```

Each target must have:

- A caption with the same filename stem
- One image with the same stem in every configured reference directory
- An edit instruction as its caption, rather than a description of the target

For example, `targets/0001.png` requires:

```text
targets/0001.txt
sources/0001.png
sources_b/0001.png      # Required only when sources_b is configured
```

## Dataset Configuration

Create `D:/datasets/krea2-edit/dataset.toml`:

```toml
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "D:/datasets/krea2-edit/targets"
reference_directories = [
    "D:/datasets/krea2-edit/sources",
    # "D:/datasets/krea2-edit/sources_b",
]
cache_directory = "D:/datasets/krea2-edit/cache"
num_repeats = 1
```

Uncomment `sources_b` when training with two references.

## Model Requirements

The workflow requires three model files:

- **DiT:** Krea 2 RAW `raw.safetensors`
- **VAE:** Qwen-Image VAE safetensors
- **Text encoder:** Qwen3-VL-4B-Instruct single-file safetensors

Train against the Krea 2 RAW checkpoint, not the Turbo checkpoint.

## PowerShell Setup

Open PowerShell in the musubi-tuner repository:

```powershell
Set-Location "D:\repos\musubi-tuner"

$Dataset = "D:\datasets\krea2-edit\dataset.toml"
$Vae = "D:\models\qwen_image_vae.safetensors"
$TextEncoder = "D:\models\qwen3vl_4b_bf16.safetensors"
$Dit = "D:\models\Krea-2-Raw\raw.safetensors"
$Output = "D:\models\training-output"
```

Replace these paths with the locations of your dataset and model files.

## 1. Cache Latents

Cache the target image and fitted reference-image latents:

```powershell
python .\krea2_edit_cache_latents.py `
    --dataset_config $Dataset `
    --vae $Vae `
    --batch_size 1 `
    --skip_existing
```

The cache command validates that every target has one or two ordered, stem-matched references.

## 2. Cache Grounded Text Outputs

Cache the image-grounded edit instructions using Qwen3-VL:

```powershell
python .\krea2_edit_cache_text_encoder_outputs.py `
    --dataset_config $Dataset `
    --text_encoder $TextEncoder `
    --text_encoder_dtype bfloat16 `
    --grounding_pixels 768 `
    --batch_size 1 `
    --skip_existing
```

`--grounding_pixels 768` creates a fixed-scale semantic-conditioning cache. If the grounding scale is changed later, incompatible caches are rebuilt rather than silently reused.

To measure grounding VRAM while processing the first batch, optionally add:

```powershell
--profile_grounding_memory
```

## 3. Train the LoRA

Launch training with Accelerate:

```powershell
accelerate launch `
    --num_cpu_threads_per_process 1 `
    --mixed_precision bf16 `
    .\krea2_edit_train_network.py `
    --dit $Dit `
    --vae $Vae `
    --dataset_config $Dataset `
    --sdpa `
    --mixed_precision bf16 `
    --timestep_sampling krea2_shift `
    --weighting_scheme none `
    --optimizer_type adamw `
    --learning_rate 1e-4 `
    --gradient_checkpointing `
    --max_data_loader_n_workers 2 `
    --persistent_data_loader_workers `
    --network_module networks.lora_krea2 `
    --network_dim 32 `
    --network_alpha 32 `
    --max_train_epochs 16 `
    --save_every_n_epochs 1 `
    --seed 42 `
    --output_dir $Output `
    --output_name krea2-edit-lora
```

The resulting LoRA checkpoints will be written under the configured output directory.

## Scaled FP8 Training

If BF16 does not fit in available VRAM and the GPU supports scaled FP8, add both of these options to the training command:

```powershell
--fp8_base `
--fp8_scaled `
```

Both flags are required. Plain `--fp8_base` without `--fp8_scaled` is rejected for Krea 2.

## Important Constraints

- The dataset batch size must be `1`.
- Each target requires one or two references.
- Reference order follows the order of `reference_directories`.
- Captions must be editing instructions.
- Latent and grounded-text caches must both exist before training.
- Use `networks.lora_krea2` as the network module.
- Use the Krea 2 RAW DiT for training.
- Do not pass `--sample_prompts`; edit-aware training previews are not implemented.
- Flip augmentation is unsupported because it would desynchronize targets and references.
- Use gradient accumulation when an effective batch size greater than one is needed.

## Rebuilding Caches

Remove `--skip_existing` when intentionally rebuilding all caches.

Caches should be regenerated after changing:

- Target or reference images
- Target resolution or bucket settings
- Reference-directory order
- Grounding scale
- Cache schema or fit-protocol version
