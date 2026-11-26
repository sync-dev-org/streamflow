# Using streamflow

This reference describes **v0.4.398**, the native C++ application. streamflow selects an input module, inference, and an output module through command arguments and an optional YAML configuration file. Each process handles one stream on one GPU. The native prefix includes LMS and NDI input/output, CDA-VSR and SPANv2. SRT input/output and SDI are not available in the standard distribution. See the [installation guide](INSTALL.md) for prerequisites and installation, [ENGINE.md](ENGINE.md) for checkpoint requirements, and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the measured scope and operational limitations.

Obtain the deb or archive, a compatible checkpoint, and `SHA256SUMS` as files from your provider. Verify the received files as described in [INSTALL.md](INSTALL.md). The deb places the application in `/opt/streamflow`, callable as `/usr/bin/streamflow`; the archive provides `prefix/bin/streamflow`. In the commands below, replace `streamflow` with the absolute path to that archive executable when using an archive. Neither form requires a Python environment, uv, or a CUDA Toolkit installation to run. TensorRT and the NVIDIA driver are host prerequisites.

## Startup arguments

Start a pipeline with arguments (replace the ID and checkpoint path with actual values):

```bash
streamflow --input-type lms --input-option connection.id=abc-defg \
  --input-option "connection.password=''" --input-option connection.channel=1 \
  --inference-type spanv2 --inference-weights /path/to/spanv2-x2.safetensors \
  --inference-option input_width=960 --inference-option input_height=540 \
  --output-type ndi --output-option source_name=streamflow --output-width 1920 --output-height 1080 --output-audio-enabled
```

| Argument | Configuration key | Meaning |
|---|---|---|
| `--input-type TYPE` | `input.type` | Input module (required). |
| `--output-type TYPE` | `output.type` | Output module (required). |
| `--inference-type TYPE` | `inference.type` | Inference module (required). |
| `--inference-weights PATH` | `inference.weights` | Checkpoint path (required). A relative argument path starts at the current directory. |
| `--output-width WIDTH`, `--output-height HEIGHT` | `output.width`, `output.height` | Positive output dimensions (required). |
| `--output-audio-enabled` | `output.audio.enabled` | Enable audio output. |
| `--pre-buffer-capacity-frames N` | `pipeline.pre_buffer_capacity_frames` | Positive pre buffer capacity in frames; default `4`. |
| `--post-buffer-capacity-frames M` | `pipeline.post_buffer_capacity_frames` | Positive capacity M in waiting frames, at least N; omitted: N + 1. |
| `--output-delay-frames N` | `output.delay_frames` | Positive receipt delay in frame periods; default `2`. |
| `--cuda-device N` | `pipeline.cuda_device` | Non-negative GPU index visible to the process; default `0`. |
| `--input-option KEY=VALUE` | `input.KEY` | Input setting; repeat for multiple settings. |
| `--output-option KEY=VALUE` | `output.KEY` | Output setting; repeat for multiple settings. |
| `--inference-option KEY=VALUE` | `inference.KEY` | Inference setting; repeat for multiple settings. |
| `--config CONFIG` | — | Optional YAML preset. |
| `--stage-trace-path PATH` | `diagnostics.stage_trace.path` | Write processing boundary timestamps to this file at shutdown. |
| `--stage-trace-capacity N` | `diagnostics.stage_trace.capacity` | Positive recording capacity; default `9000`. See diagnostics below. |
| `-h`, `--help` | — | Display usage and exit. |

Use dotted option keys such as `connection.channel`. Values follow the same YAML scalar types as the configuration file: `connection.channel=3` is an integer, `audio.enabled=true` is a boolean, and `"connection.password=''"` supplies an empty string. Shell quotes group an argument; retain YAML quotes inside it to force a string, for example `--output-option "source_name='true'"`. `--input-type=lms` and other `--option=value` spellings are also accepted. Unknown arguments are rejected.

Arguments override the same settings in `--config`, regardless of the location of `--config` in the command. For repeated assignments to the same key, the last argument wins. Omitted settings use the defaults below. Every required setting must come from arguments or the file, including connection settings required by the selected module. All configuration settings can also be supplied through arguments. Copy [settings-example.yaml](settings-example.yaml) to `settings.yaml` for a complete preset, then run `streamflow --config settings.yaml`, or override part of it with `streamflow --config settings.yaml --output-width 1280 --output-height 720`.

The configuration is read at startup. Stop and restart the application to apply edited settings. Windows batch files can also assemble these arguments; see [INSTALL.md](INSTALL.md#windows-batch-files-for-wsl).

## Configuration file

The top-level sections are `pipeline`, `input`, `inference`, `output`, and the optional `diagnostics` section. Any section can be omitted from the file when its required settings are supplied by arguments. Items marked "Required" in the tables have no defaults. streamflow rejects unknown sections and unknown items for the standard modules. Additional modules receive their own settings as dotted keys and string values through the core; installing such modules is described below.

Item names in the tables are relative to each section. Use integers for integer settings, `true` / `false` for booleans, and a `"numerator/denominator"` string such as `"30000/1001"` for fractions. streamflow checks the types of supplied values and passes values such as connection settings and video formats to each companion for validation. Leave omitted items out of the file; do not specify `null`.

### `pipeline`

| Item | Type | Default | Meaning |
|---|---|---|---|
| `cuda_device` | Non-negative integer | `0` | GPU index visible to the process. |
| `pre_buffer_capacity_frames` | Positive integer | `4` | Capacity of the pre buffer between the input module and thread P, in frames. |
| `post_buffer_capacity_frames` | Positive integer | `output.delay_frames + 1` | Capacity M in waiting frames, at least N. The last image retained for repeats is not counted. |

Use `--pre-buffer-capacity-frames`, `--post-buffer-capacity-frames`, and `--cuda-device` to set these values at startup, for example `streamflow --config settings.yaml --pre-buffer-capacity-frames 8 --post-buffer-capacity-frames 2 --cuda-device 0`. Arguments use the same value checks as the configuration file; invalid values stop startup with exit code `2` before the first frame.

streamflow creates and owns both drop buffers. When full, each buffer drops the oldest frame together with its audio. Thread P takes frames from the pre buffer, runs inference and conversion to the output resolution, waits for GPU completion, and passes the result to the post buffer. The output module controls the send cadence and repeats the last image with silence when input is insufficient. `pipeline.exit_fifo_depth` is not an accepted setting.

The receipt delay N sets latency. Increasing capacity M above N keeps more early arrivals without changing that delay. A capacity below N stops startup and reports both settings and values.

### `input`

| Item | Type | Default | Meaning |
|---|---|---|---|
| `type` | String | Required | `lms` or `ndi`. |

#### LMS input

For `type: lms`, put connection settings in the `connection` mapping. streamflow passes the following settings to the LMS input module.

| Item | Type | Default | Meaning |
|---|---|---|---|
| `connection.id` | String | Required | LMS connection ID. Exactly 8 characters: three lowercase letters, a hyphen, and four lowercase letters (for example `abc-defg`). |
| `connection.password` | String | Required | LMS connection password. At most 8 ASCII characters; an empty value is allowed. Spaces and symbols are passed unchanged. |
| `connection.channel` | Integer `1..255` | Required | LMS channel. The channels a connection ID may use are set by that ID's LMS contract, and video and audio can allow different numbers of channels. For a video channel above the allowance, the LMS service accepts the connection and then reports `Channel has exceeded the allowed number!`; no video is delivered. |
| `fallback_frame_rate` | Positive fraction string | `"30000/1001"` | Fallback frame rate in frames/s when the HEVC VUI has no timing or its period is outside the eight values accepted by LMS input. |
| `decode_capacity_frames` | Integer `1..65535` | `30` | Depth of the blocking buffer that holds encoded access units before decoding, and the number of decoder surfaces. |
| `audio_trim_unit_samples` | Integer `0..48000` | `240` | Number of sample frames in one unit of audio trimmed by the receiving SDK, before multiplying by the channel count. `0` excludes the trimmed amount from the audio position. |
| `receive_delay_ms` | Non-negative integer | `120` | LMS receive latency in ms: how long the LMS SDK waits on the receiving side. |

`decode_capacity_frames` is the capacity for encoded access units arriving while decoding falls behind. The default of 30 is about 1 second at 29.97 fps. When decoding keeps up, the buffer is empty, so increasing its depth does not increase latency. Set the depth of the pre buffer that drops frames after decoding separately with `pipeline.pre_buffer_capacity_frames`. The LMS vendor SDK runs inside the streamflow process; if the SDK terminates abnormally, the streamflow process stops with it.

LMS input accepts VUI periods of `1/24`, `1001/24000`, `1/25`, `1/30`, `1001/30000`, `1/50`, `1/60`, and `1001/60000` seconds. It uses the reciprocal of `input.fallback_frame_rate` as the fallback period when VUI timing is absent or outside this set.

The default `audio_trim_unit_samples` value of 240 corresponds to 5 ms at 48 kHz. This default of 240 comes from observations with the pinned SDK and is not guaranteed by the vendor. streamflow passes the value unchanged to the LMS input module.

LMS input has no color fallback or override settings. The core's entrance conversion fills missing color information with legal, BT.709, and left.

```yaml
input:
  type: lms
  connection:
    id: "<LMS_ID>"
    password: "<LMS_PASSWORD>"
    channel: 1
  fallback_frame_rate: "30000/1001"
  decode_capacity_frames: 30
  audio_trim_unit_samples: 240
  receive_delay_ms: 120
```

LMS input rejects unknown settings before the first frame and reports their location.

#### NDI input

Select `--input-type ndi` or `input.type: ndi`. The standard distribution includes the NDI input module. Set `source_name` to the source name reported by the NDI sender, including any host prefix and spaces.

| Item | Type | Default | Meaning |
|---|---|---|---|
| `source_name` | Non-empty string without NUL | Required | NDI source to receive. Spaces and symbols are passed unchanged. |
| `receive_delay_ms` | Integer `0..2147483647` | `120` | Receive delay in ms, applied equally to video and audio by the NDI SDK. `0` disables the additional wait. |

Arguments override the file; when neither supplies `receive_delay_ms`, streamflow passes 120 ms. Fixed receive-side waits configured by input modules are excluded from the 100 ms latency target. Invalid names or delays stop startup before the first frame and identify the setting and value.

```yaml
input:
  type: ndi
  source_name: "camera (Studio A)"
  receive_delay_ms: 120
```

For example, receive NDI, process it with SPANv2, and send a different NDI source:

```bash
streamflow --input-type ndi --input-option "source_name='camera (Studio A)'" \
  --input-option receive_delay_ms=120 --inference-type spanv2 --inference-weights /path/to/spanv2-x2.safetensors \
  --output-type ndi --output-option source_name=streamflow --output-width 1920 --output-height 1080 --output-audio-enabled
```

NDI input does not provide motion vectors or residuals. It cannot be combined with CDA-VSR; streamflow reports the selected input and inference and exits with code `2` before processing the first frame. Use a compatible SPANv2 checkpoint for NDI input. The input receives frames in arrival order without frame sync; output cadence remains controlled by the output module.

SRT input is not supported by this version of streamflow. Selecting `input.type: srt` reports the unsupported input setting and exits before the first frame.

### `inference`

| Item | Type | Default | Meaning |
|---|---|---|---|
| `type` | String | Required | `cda-vsr` or `spanv2` (`type: spanv2` selects the SPANv2 module). |
| `weights` | String | Required | Path to a safetensors checkpoint. Relative paths are resolved from the configuration file's location. |
| `precision` | String | `fp16` | `fp16` or `fp32`. |
| `input_width` | Positive integer | Width of the first frame | Width of frames passed to inference. |
| `input_height` | Positive integer | Height of the first frame | Height of frames passed to inference. |

If both `input_width` and `input_height` are specified, streamflow opens the model before starting input reception, reducing the delay to the first output caused by frames accumulating immediately after startup.

streamflow rejects frames whose size differs from the inference input size determined at startup. Obtain the model checkpoint separately; the archive contains no weights or engines. The companion handles engine building and caching. CDA-VSR accepts producer-exported checkpoints trained with luma-range residuals and metadata `cda_vsr.residual_unit=luma-range`; checkpoints without this metadata are rejected at startup. Format conversion or manually adding metadata does not establish compatible training or replace the producer's export. SPANv2 requires a compatible SPANv2 safetensors checkpoint. `inference.weight_prefix` is not accepted for any model. See [ENGINE.md](ENGINE.md) for checkpoint preparation and for cache locations.

There is no operator command to exchange the input, output, or model in a running CLI process. Stop and restart streamflow to use a different selection or settings.

### `output`

The following items are common to outputs. Item names in the table are relative to output.

| Item | Type | Default | Meaning |
|---|---|---|---|
| `type` | String | Required | `lms` or `ndi`. |
| `width` | Positive integer | Required | Width of images placed in the post buffer. Images are resized if the inference result has a different size. |
| `height` | Positive integer | Required | Height of images placed in the post buffer. Images are resized if the inference result has a different size. |
| `delay_frames` | Positive integer | `2` | Receipt delay N; storage capacity M must be at least N. |
| `prefill_frames` | Integer `1` for NDI and LMS | `1` | NDI and LMS require one initial frame for receipt scheduling. Other values stop startup before the first frame; choose the delay with `--output-delay-frames`. |
| `audio.enabled` | Boolean | `false` | Pass input audio together with the image only when `true`. |
| `audio.max_wait_ms` | Finite non-negative number | One output frame period, in ms | Maximum wait in milliseconds for a frame's audio immediately before output transmission, shared by NDI and LMS. At the limit, missing audio is filled with silence. `0` sends immediately with silence for any missing audio. |

When neither arguments nor the configuration file supplies `output.audio.max_wait_ms`, streamflow passes one frame period in milliseconds, using the period with which the output opens: about 33.367 ms at 30000/1001 fps, 40 ms at 25 fps, or 16.683 ms at 60000/1001 fps. When the input period changes and the output reopens, the default follows the new period. An explicit value, including `0`, stays unchanged. Negative and non-numeric values stop startup before the first frame and identify the setting and value. Use `--output-option audio.max_wait_ms=0.125` to override the configuration file. Input reception and inference do not wait for audio; the output fills any missing audio with silence when the wait expires and sends the accepted frame.

NDI and LMS use the core's `receipt_schedule` mode. `--output-delay-frames` or `output.delay_frames` selects delay N (default 2). `--post-buffer-capacity-frames` or `pipeline.post_buffer_capacity_frames` selects storage capacity M (default N + 1, minimum N). The core selects each image D = N × one input frame period after input receipt, then encodes and sends it. M − N extra slots retain early arrivals without changing D; the last image retained for repeats does not occupy these waiting slots. Other exit modes do not receive N. D stays fixed during a run and excludes encoding and transmission after selection; at 30000/1001 fps, N = 2 gives about 66.7 ms.

The core follows input clock drift using the receipt times and input frame numbers. streamflow uses the core's default phase adjustment limit. Processing completed within D causes neither a post buffer drop nor a repeated image. An image that misses its selection time is discarded with its audio; that output period repeats the last image with silence. After an input restart, input number scope change, or frame period change, the new first receipt establishes the same N-period delay.

streamflow creates and opens the output module after receiving the first frame, using that frame's `timing.input_number.frame_period`. When a later frame has a different period, streamflow waits for the previous output module to send its remaining frames, closes it, and opens a new output module and post buffer for the new period before forwarding that frame. Output is interrupted while the module is reopened. Frames with an unchanged period use the same output module; streamflow does not convert frame rates. `output.frame_rate` cannot be specified. For NDI and LMS, streamflow validates `output.prefill_frames=1` before receiving the first frame.

#### NDI output

For `type: ndi`, set the source name in `source_name`. It must be a non-empty string and defaults to `streamflow`. Spaces and symbols in the supplied string are not changed. NDI output does not accept `encoded_depth`; specifying it in YAML or with `--output-option encoded_depth=...` stops startup before the first frame and identifies `output.encoded_depth`.

```yaml
output:
  type: ndi
  source_name: streamflow
  width: 1920
  height: 1080
  prefill_frames: 1
  audio:
    enabled: false
```

#### LMS output

For `type: lms`, put connection, video, encoder, and conversion settings in separate mappings. Put width and height directly under output, not inside video.

LMS output does not accept `encoded_depth`; specifying it in YAML or with `--output-option encoded_depth=...` stops startup before the first frame and identifies `output.encoded_depth`.

LMS output always uses 0 for the acquisition timestamp passed to the SDK.

| Item | Type | Default | Meaning |
|---|---|---|---|
| `connection.id` | String | Required | LMS connection ID. Exactly 8 characters: three lowercase letters, a hyphen, and four lowercase letters (for example `abc-defg`). |
| `connection.password` | String | Required | LMS connection password. At most 8 ASCII characters; an empty value is allowed. |
| `connection.channel` | Integer `1..255` | Required | LMS channel. The channels a connection ID may use are set by that ID's LMS contract, and video and audio can allow different numbers of channels. For a video channel above the allowance, the LMS service accepts the connection and then reports `Channel has exceeded the allowed number!`; no video is delivered. |
| `video.bit_depth` | Integer | `8` | Output bit depth. |
| `video.colorspace` | String | `Rec.709` | Output colorspace. |
| `video.gamma` | String | `Rec.709` | Output gamma. |
| `video.matrix` | String | `BT.709` | Output matrix. |
| `video.siting` | String | `left` | Output chroma siting. |
| `video.range` | String | `legal` | Output range. |
| `encoder.bitrate_bps` | Integer | `10000000` | Encoding bitrate in bit/s. |
| `encoder.gop_frames` | Integer | `30` | GOP length in frames. |
| `encoder.idr_interval_frames` | Integer | Value of `gop_frames` | IDR period in frames. |
| `encoder.rate_control` | String | `cbr` | `cbr` or `vbr`. |
| `encoder.max_bitrate_bps` | Integer | `0` | Maximum bitrate passed to the encoder, in bit/s. |
| `conversion.allow_colorspace` | Boolean | `true` | Allow colorspace conversion in the output module. |
| `conversion.allow_gamma` | Boolean | `true` | Allow gamma conversion in the output module. |
| `conversion.allow_matrix` | Boolean | `true` | Allow matrix conversion in the output module. |

```yaml
output:
  type: lms
  connection:
    id: "<LMS_ID>"
    password: "<LMS_PASSWORD>"
    channel: 1
  width: 1920
  height: 1080
  prefill_frames: 1
  encoder:
    rate_control: cbr
    max_bitrate_bps: 0
  conversion:
    allow_colorspace: true
    allow_gamma: true
    allow_matrix: true
  audio:
    enabled: false
```

LMS output does not accept flat keys (`output.id`, `output.bitrate_bps`, `output.video_range`, `output.work_queue_depth`, and so on). `output.encoder.vbv_buffer_bits`, `output.encoder.latency_profile`, and `output.buffering.work_queue_depth` are also rejected. Unknown keys are rejected before the first frame, with their location reported.

#### SRT output

SRT output is not supported by this version of streamflow. Selecting `output.type: srt` reports the unsupported output setting and exits before the first frame.

### Additional modules and startup checks

The standard components are included in the native prefix. Additional components must be supplied as compatible native distributions placed in the same prefix layout; Python extras and wheels do not add modules to this application. A missing module, a module that cannot be loaded, and an incompatible module ABI produce distinct startup diagnostics. Read the selected component name, the core's reason, and any distribution name or installation guidance in the error. Missing selected components stop startup before the first frame.

streamflow checks the selected modules and the input/model combination during startup. A model that needs motion vectors and residuals requires an input that supplies them. Checkpoint or engine preparation errors stop startup with exit code `2`; the model may be prepared before reception when both input dimensions are supplied, or when the first frame arrives otherwise. Output checks that depend on the actual frame period happen when the output is opened for that frame. Correct the reported setting or component and start again.

### `diagnostics.stage_trace`

Recording is disabled unless a path is supplied. Use `--stage-trace-path trace.json` or the following optional configuration:

```yaml
diagnostics:
  stage_trace:
    path: trace.json
    capacity: 9000
```

`path` must be a non-empty string. A relative trace path starts at the process's current directory, including when written in a configuration file. The application opens and truncates the file at startup; its parent directory must exist and be writable. `capacity` must be a positive integer and defaults to `9000`, even when recording is disabled.

Recording retains recent timestamps in bounded input, processing, and output series, overwriting older records when capacity is reached. The processing series retains up to `capacity` recent frames; input and output series can overwrite earlier when there are repeated events within a frame. streamflow writes one JSON document to the trace file after cleanup, with the format name `streamflow.stage_trace`, version `1`, and `CLOCK_MONOTONIC` timestamps in nanoseconds. Standard output remains JSON lines. A second stop request can exit before the trace document is written. A trace write failure is reported on standard error and returns exit code `1`.

## Reading standard output

streamflow writes progress, operational events, and final statistics to standard output as one JSON object per line. No verbose option is needed. Argument and startup configuration errors are written to standard error with a description beginning with `streamflow:`.

| Item | Meaning |
|---|---|
| `kind` | `progress` for progress, `event` for operational events, and `summary` for final statistics. |
| `elapsed_s` | Seconds since the CLI began recording operational events, including connection and engine preparation time. |

### `progress` and `summary`

streamflow emits progress approximately every 0.05 seconds, including when no input is available or while waiting for modules or inference to start.

| Item | Records | Meaning |
|---|---|---|
| `received` | Both | Number of frames returned by the input module's `next()`. Images dropped by the pre buffer are not included. |
| `processed` | Both | Number of images that completed inference, conversion, and the wait for GPU completion. |
| `sent` | Both | Sum of `inspection.sent_periods` across output modules, including modules closed after a period change. Includes send periods that repeat the last image; it is not a count of arrivals at the receiver. |
| `frame_selection` | Summary only | Frame selection mode reported by the last output (`receipt_schedule` for NDI/LMS). Null if no output was created. |
| `schedule_scope` | Summary only | Opaque scope identity hash for the last output's receipt grid; null before a grid is anchored. It is local to the process and is not a cross-run frame identifier. |
| `phase_correction_ms` | Summary only | Phase correction reported by the last output, in milliseconds; null if no output was created. |
| `receipt_error_ms` | Summary only | Last observed receipt time minus the grid's expected receipt time, including its phase before the current correction. The core retains this observation on repeats and clears it when a new grid is anchored. Null when unobserved. |
| `pre_dropped` | Both | Number of images dropped because the pre buffer was full. |
| `post_dropped` | Both | Total post buffer drops from overflow and expired selection times, summed across buffers replaced after period or output changes. |
| `post_expired` | Both | Subset of `post_dropped` discarded because their selection times expired. The discarded images and their audio are not sent. |
| `dropped` | Both | Sum of `pre_dropped` and `post_dropped`. |
| `faulty_reference` | Both | Number of frames passed to inference with the missing-reference flag set to true. Explicit false and an absent flag are excluded. |
| `faulty_reference_unobserved` | Both | Number of frames passed to inference without an observed missing-reference flag. An absent flag is unknown, rather than a false observation. |
| `faulty_reference_processed` | Both | Number of frames with the missing-reference flag set to true that completed inference, conversion, and the wait for GPU completion, and were accepted by the post buffer. Excludes inference failures and post buffer acceptance failures. |
| `faulty_reference_processed_unobserved` | Both | Number of frames without an observed missing-reference flag that completed inference, conversion, and the wait for GPU completion, and were accepted by the post buffer. Excludes inference failures and post buffer acceptance failures. |
| `faulty_reference_post_dropped` | Both | Number of images dropped by the post buffer whose missing-reference flag was confirmed to have been true using the input number's scope and number recorded at insertion. |
| `faulty_reference_post_dropped_unobserved` | Both | Number of images dropped by the post buffer whose missing-reference flag was absent, or whose insertion record could not be matched (for example, because they had no input number). Explicit false does not increase this counter. |
| `send_fps` | Both | In progress records, the increase in `sent` over the preceding second; the count at the start of that second is linearly interpolated between observations. During the first second, the count since startup is used without extrapolation; zero before any sends are observed. In statistics, `sent / elapsed_s`. |
| `frame_ms` | Summary only | Thread P's frame processing time in milliseconds: `count`, `mean`, `median`, `p90`, and `maximum`. See below. |
| `age_ms` | `progress` | Milliseconds from the last processed frame's `next()` return to immediately before passing it to the post buffer. `"unobserved"` if not observed. |

`age_ms` retains the same value until the next frame is processed. It excludes time queued in the pre/post buffers and subsequent transmission; initialization or output opening performed while processing that frame can be included. It is not end-to-end latency.

`frame_ms` measures from just after thread P takes a frame from the pre buffer until the GPU completion wait returns. It includes processing and any initialization performed for that frame, and excludes pre/post buffer waits and transmission. Only frames that complete processing contribute. With no samples, `count` is `0` and the other fields are `null`. The median and p90 are estimates from a bounded histogram; the mean and maximum use the observed durations. These are elapsed durations, not GPU wait thread CPU time or end-to-end latency. This object does not appear in progress records.

A zero `faulty_reference` count means that no observed flag was true; it does not prove that every frame was examined. Read `faulty_reference_unobserved` alongside it. The processed and post-drop counters use the same distinction between true, false, and unobserved.

Post buffer insertion records retain only recent input numbers and missing-reference flags, bounded by the buffer depth. They do not retain image or side data arrays. The side data of images dropped by the pre buffer never reaches streamflow, so their missing references cannot be counted. To check that frames with missing references progressed through processing, confirm `pre_dropped=0`, `faulty_reference>0`, and `faulty_reference=faulty_reference_processed`, and also read the post buffer counts for missing-reference drops and unobserved drops. Acceptance by the post buffer does not indicate arrival at an external receiver.

### `event`

| `event` value | Additional fields | Meaning |
|---|---|---|
| `config.duplicate_key` | `path` | Location of a duplicate key in the same YAML mapping. The later value is used, and a warning is emitted once before the first frame. |
| `startup.opening` | `role` | Starts opening the input (`entrance`) or output (`exit`) connection. |
| `startup.opened` | `role` | The input (`entrance`) or output (`exit`) connection's `open()` has returned successfully. Not emitted while retrying or when stopping. |
| `startup.waiting` | `role`, `error_type`, and `error_reason` if available | Waits after a recoverable failure during opening. |
| `input.colour` | `observed`, `defaults_used`, `resolved` | Records colors when an input snapshot is observed. The current core delivers the entrance snapshot with the first image, so this event normally accompanies the first image. During operation, colors are recorded again only when they change. |
| `input.first_frame` | `sequence`, `input_number` | Records the sequence and input number of the first image returned by `next()` once, before inference. Does not substitute the initial number from the open snapshot. |
| `input.frame_period` | `sequence`, `input_number`, `frame_period`, `initial_frame_period` | For the first image and images whose period changes, records the actual `timing.input_number.frame_period` and the initial period from the open snapshot as fraction strings. |
| `output.reopened` | `previous_frame_period`, `frame_period` | The output module finished sending frames for the previous period and was reopened for the new period. Both periods are fraction strings. |
| `input.interrupted` | `error_type`, `error_message`, and `error_reason` if available | `next()` returned a recoverable failure. Emitted once at the start of an interruption. |
| `input.reconnected` | None | Processing of the frame after an interruption has completed. |
| `exit.recoverable` | `error_type`, `error_message`, and `error_reason` if available | Reports a recoverable failure in the output module and continues with the same output. |
| `drop` | `sequence`, `input_number`, `reason` | A drop record from the core. The reason is `pre-buffer-full`, `post-buffer-full`, or `post-buffer-expired`. Records without an input number use `null` for `input_number`. |
| `inference.reset` | `sequence`, `causes` | Resets CDA-VSR according to the input epoch, continuity scope, and source discontinuity flag. |
| `fatal` | `reason`, `error_type`, and `error_reason` if available | Reports a failure that prevents continuation once, then stops. |
| `stopping` | None | Starts cleanup in response to a stop request. |
| `stop.aborted` | None | A second or subsequent stop request aborted outstanding waits. |

CDA-VSR also resets for the first frame, the first frame after an interruption, and HEVC I frames (`codec=hevc`, `slice_type=2`). Frames with the missing-reference flag set to true are also passed to inference and sent to the output. Input timing and ancillary data are preserved in the output frame. With audio enabled, the frame retains its audio reference through inference; the output module resolves the audio immediately before transmission, subject to `output.audio.max_wait_ms`.

`input.colour.observed` contains `range`, `matrix`, `colorspace`, `gamma`, and `chroma_siting`, using the color names returned by the core and `null` for unobserved values. `defaults_used` contains the fields and values that were filled in; `resolved` contains all resolved color fields. If no fields were filled in, `defaults_used` is `{}`. If the snapshot itself has no color information, all three fields are `null`, distinguishing this case from one where no defaults were needed.

The following example shows records where the core supplies colors because no color values were observed, and the initial period at open differs from the actual image period. `elapsed_s` is omitted for this example.

```json
{"kind":"event","event":"input.colour","observed":{"range":null,"matrix":null,"colorspace":null,"gamma":null,"chroma_siting":null},"defaults_used":{"range":"legal","matrix":"BT.709","colorspace":"Rec.709","gamma":"Rec.709","chroma_siting":"left"},"resolved":{"range":"legal","matrix":"BT.709","colorspace":"Rec.709","gamma":"Rec.709","chroma_siting":"left"}}
{"kind":"event","event":"input.first_frame","sequence":241,"input_number":241}
{"kind":"event","event":"input.frame_period","sequence":241,"input_number":241,"frame_period":"1001/30000","initial_frame_period":"1/25"}
```

## Shutdown and exit codes

To stop, press Ctrl+C (SIGINT) once or send SIGTERM to the process once. streamflow asks the input module to stop, waits for accepted inference and GPU work to finish, lets the output module finish processing the remaining post buffer contents, closes both modules, and finally emits `summary`.

A second or subsequent SIGINT or SIGTERM requests that streamflow abort outstanding waits for inference opening and execution, and input and output opening, join, and close. If waits are aborted, streamflow emits `stop.aborted` and `summary` and exits with code `1`. If no waits are aborted, normal shutdown behavior applies.

| Exit code | Meaning |
|---|---|
| `0` | Normal shutdown completed. Stopping a connection wait during startup or displaying `--help` also returns `0`. |
| `1` | A failure prevented continuation during operation or shutdown, or a second or subsequent stop request aborted completion waits. |
| `2` | Operation could not start because of a problem with arguments, the configuration file, a companion, a checkpoint, startup processing, or similar. The reason is written to standard error. |

When a startup error results in exit code `2`, streamflow does not emit `summary`. Read the item name and reason on standard error, correct the configuration or runtime environment, and start again.

Copyright © 2026 minamikik / sync.dev. All rights reserved.
