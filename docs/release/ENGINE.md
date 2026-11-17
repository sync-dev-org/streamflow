# CDA-VSR and SPANv2 checkpoints and engine cache

This guide describes **v0.4.398**, the native C++ application. The deb and archive contain no model checkpoints, weights, or TensorRT engines. Receive a checkpoint supported by the selected inference module and its `SHA256SUMS` file from your provider, and verify the files as described in [INSTALL.md](INSTALL.md).

Pass the safetensors checkpoint path with `--model PATH` or set `inference.model` in YAML. Relative argument paths start at the process's current directory; relative YAML model paths start at the configuration file's directory. Do not supply an engine path. The companion builds missing engines and manages the cache.

## Included component versions

The v0.4.398 native prefix uses the following component versions. TensorRT remains a host prerequisite; see [INSTALL.md](INSTALL.md).

| Component | Version |
|---|---|
| pixtreme | 2.27.0 |
| Transport core | 0.36.0 |
| LMS input/output | 0.29.2 |
| NDI output | 0.28.0 |
| NVDEC | 0.11.2 |
| NVENC | 0.10.0 |
| Inference core | 0.26.0 |
| CDA-VSR | 0.19.0 |
| SPANv2 | 0.9.0 |
| tensoracle | 2.6.1 |

## Prepare a CDA-VSR checkpoint

CDA-VSR requires a checkpoint trained with residuals normalized by the luma black-to-white range and exported by the checkpoint producer. The safetensors header metadata must contain `cda_vsr.residual_unit` with the exact string value `luma-range`.

Supply a checkpoint meeting these requirements. Checkpoints without `cda_vsr.residual_unit=luma-range` are rejected at startup. Converting a file format, extracting `params_ema`, or manually adding metadata does not establish compatible training or replace the producer's export. The export must apply any training-specific weight adjustments. Preserve all producer-written metadata when copying the checkpoint.

The CDA-VSR companion checks the following before opening an engine:

- `cda_vsr.residual_unit` must be `luma-range`. Absent metadata, a missing key, or any other value is rejected. streamflow reports checkpoint or inference preparation failures as startup errors on standard error and exits with code `2`.
- `cda_vsr.gate`, when present, must be `sigmoid`. An absent gate key selects the `1 + w` gate. Preserve the producer's choice; other values are rejected before reading tensor values or writing a cache.
- Exactly one weight set must be present. The module recognizes sets by the documented tensor names under a common prefix such as `params.` or `params_ema.`, or without a prefix. Multiple sets, incomplete additional sets, or no sets are rejected with the detected set names. Supply the single inference set selected and exported by the producer.
- The header shape of `upconv1.weight` determines the positive integer scale. Engine building additionally validates the tensor names, shapes at that scale, and float32 dtypes. A successful header query alone does not establish that all weights are valid.

For a compatible checkpoint, select CDA-VSR as follows. Replace the path with the producer's exported file:

```yaml
inference:
  arch: cda-vsr
  model: models/cda-vsr.safetensors
  precision: fp16
```

CDA-VSR requires an input module that supplies motion vectors and residuals. LMS input provides this data. streamflow rejects an incompatible input/model combination before taking the first frame.

## Prepare a SPANv2 checkpoint

Supply a compatible producer-exported SPANv2 safetensors checkpoint and select
`inference.arch: spanv2`. The standard native prefix includes this module; the inference runtime does not require PyTorch.

```yaml
inference:
  arch: spanv2
  model: models/spanv2.safetensors
  precision: fp16
```

`inference.weight_prefix` is not accepted for any model.

## Engine preparation and cache

The standard prefix includes CDA-VSR, SPANv2 and tensoracle. The host supplies TensorRT; see [INSTALL.md](INSTALL.md). Provide a compatible NVIDIA GPU and driver/runtime for CUDA 13. At startup, each companion selects compatible cached engines or builds missing engines from the checkpoint. Engine building takes time and requires the GPU. Checkpoints and engines remain external to the archive.

The cache is located at `$XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle` when `XDG_CACHE_HOME` is unset. A different checkpoint, input size, precision, GPU, or TensorRT environment may require another build. CDA-VSR uses separate engines for initializing temporal state and carrying state forward. streamflow prepares the selected model on `pipeline.cuda_device` (default `0`) and the companion manages missing engines through tensoracle. Keep the cache writable by the account running streamflow; normal operation does not require root.

`inference.precision` defaults to `fp16`; `fp32` is also accepted. Set both `inference.input_width` and `inference.input_height` to the actual input size to prepare inference before starting input reception. When dimensions are omitted, streamflow determines them from the first frame and prepares inference then. Subsequent frames must match that input size. `output.width` and `output.height` set the final output size independently.

To prepare engines before reception, set both `inference.input_width` and
`inference.input_height` in the configuration to the actual incoming dimensions,
then start the native application with `streamflow --config settings.yaml`
(or `./prefix/bin/streamflow --config settings.yaml` from an archive).
Startup performs preparation and reports any checkpoint or engine errors. No
Python environment or separate Python engine-building command is required.
See [usage](USAGE.md) for configuration and shutdown, and
[known limitations](KNOWN_LIMITATIONS.md) for the measured scope.

Copyright © 2026 minamikik / sync.dev. All rights reserved.
