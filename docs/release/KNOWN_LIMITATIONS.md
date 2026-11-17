# Known limitations

This guide covers **v0.4.398**, the native C++ application. streamflow connects LMS input to CDA-VSR or SPANv2 inference and LMS or NDI output through the C++ transport core. Each process handles one stream on one GPU. See the [installation guide](INSTALL.md), [usage reference](USAGE.md), and [checkpoint requirements](ENGINE.md).

## Measured operation

Measurements apply only to the application version, checkpoint, and conditions stated. None of these observations guarantees performance on other hardware or compatibility of another checkpoint.

### Native v0.4.398: RTX A6000 under WSL2

The measurements used the native shared-library prefix on one NVIDIA RTX A6000 under WSL2, with exclusive use of that GPU and all CPU cores. The sender and receivers ran on the same host; the sender and LMS receiver also used that GPU. Each run sent 9,000 frames at 30000/1001 fps through LMS input, CDA-VSR x2 fp16 from 960x540 to 1920x1080, and NDI or LMS output, with audio enabled.

#### Sender-to-NDI-receiver latency

Latency was measured from the sender handing a frame to LMS until the NDI receiver received it, excluding the 100 ms wait configured by `input.delay_ms`. The three runs changed `output.audio.max_wait_ms` as follows:

| Audio wait limit | Samples | Minimum | Median | p90 | Maximum | First-half median | Second-half median |
|---|---|---|---|---|---|---|---|
| 0 ms | 135 | 47.1 ms | 48.5 ms | 50.0 ms | 59.9 ms | 48.0 ms | 49.4 ms |
| Default: one period, about 33.4 ms | 135 | 46.7 ms | 49.6 ms | 54.7 ms | 56.1 ms | 51.7 ms | 49.3 ms |
| 66.7 ms | 135 | 46.9 ms | 49.4 ms | 56.1 ms | 57.8 ms | 55.0 ms | 49.3 ms |

All 405 samples were below 100 ms under these same-host conditions. End-to-end latency for LMS output could not be obtained because the receiver's flashes could not be matched uniquely to the sender's flashes.

#### Audio observations

With all three wait limits, the NDI receiver recorded equal video and audio packet counts, audio packet sizes of 1601 or 1602 sample frames, and a median interval of 2002 ms between known tones, matching the source. At the LMS receiver, all audio packets except the last were complete, and no audio references were reported as unavailable.

The median interval from a received image flash to the corresponding tone onset was +6.5 ms for NDI and +13.0 ms for LMS for all three wait limits; positive values mean the audio followed the image. These observations cover the stated source and receivers. The actual time spent waiting for audio at the product's output, and the number of samples filled with silence there, are not available from standard output or the trace. The default wait limit remains one input frame period.

#### Processing and output rates

Separate rate measurements used `input.delay_ms: 0` without diagnostic trace recording. The output ratio is the observed output send rate divided by the nominal 30000/1001 fps rate; it does not measure receiver arrival rate.

| Output | Thread P processing rate | Output send rate / nominal rate |
|---|---|---|
| NDI | 29.952 fps | 0.998730 |
| LMS | 29.808 fps | 1.002656 |

**Thread P did not reach 29.97 fps with either output.** These are single-run rate measurements; the cause of the shortfall has not been established. LMS-output runs in this measurement set dropped 540–554 frames in the pre buffer per run. These results do not establish uninterrupted processing at the nominal rate.

## Historical measurements

The following results describe earlier versions and their stated conditions. The Python-runtime results do not validate the native C++ application.

### Native v0.4.396: startup, output, and shutdown

On 2026-11-16, the v0.4.396 deb was installed with apt on WSL2 / Ubuntu 24.04 on a PC with one RTX 5090. The Windows batch files started LMS input through CDA-VSR x2 and SPANv2 x2 to NDI output, and both runs stopped with exit code `0`. These two runs shared the GPU and ran concurrently. This was a functional check of startup, output, and shutdown; performance, latency, and audio/video correspondence were not measured.

### Earlier Python runtime: RTX A6000 under WSL2

The following historical connection and performance measurements were made on an NVIDIA RTX A6000 under WSL2. The application versions for these runs are not recorded in this guide; they must not be read as v0.4.398 results.

- LMS input, CDA-VSR x2 upscaling from 960x540 to 1920x1080, and NDI output ran with a Windows NDI receiver on the same PC. The receiver obtained 2,408 frames at 30000/1001 fps, with no backward or duplicate timecodes.
- For LMS input and LMS output, a separate receiver process received the same number of frames as the output module's sent periods, without gaps in received frame numbers.
- Eight synthetic H.265 streams with different properties, including B frames, absent VUI color information, absent VUI timing, IDR-only pictures, and CRA pictures, passed from LMS input to NDI output without stopping.
- The 2026-10-19 performance validation upscaled 960x540 to 1920x1080 with an x2 test checkpoint trained before the residual unit was changed to luma-range. Each run used 30,240 input frames (about 17 minutes). After consuming the startup backlog, thread P sustained 29.970 fps on this PC. These runs do not establish performance with a luma-range-trained checkpoint.

The performance runs recorded the following output counts and CPU time spent by the thread waiting for GPU completion. The CPU measurement uses `time.thread_time_ns`, in milliseconds per processed frame; it is not the elapsed duration of the wait or total process CPU utilization.

| Output | Output send periods | Received frames | GPU wait thread CPU time, median / p90 |
|---|---|---|---|
| NDI | 30,258 | 30,258 | 3.43 / 3.80 ms |
| LMS | 30,247 | 30,246 | 3.40 / 3.80 ms |

The cause of the one-frame difference between LMS send periods and received frames has not been established.

### Earlier Python runtime: RTX 5090 under WSL2 (Ubuntu 24.04)

Installation from an earlier Python-based archive was checked on an RTX 5090 (sm 120, 32 GB) under WSL2 with Ubuntu 24.04.5, glibc 2.39, and Windows NVIDIA driver 617.42. The application version of that installation check and the separate test checkout was not recorded in this guide. The engine preparation and short inference check below used **v0.4.169** with the CDA-VSR module version **0.10.2**, on 2026-10-23.

- After installing the required host build packages, the historical installation command installed streamflow and its runtime modules in a fresh CPython 3.12 environment. `streamflow --help` and imports of the LMS, NDI, and CDA-VSR modules passed. This validation did not include SPANv2. It is not an installation procedure for v0.4.398, which requires no Python environment.
- CuPy reported one visible CUDA device. In a separate repository checkout, 291 CPU tests and 78 GPU tests passed. The GPU tests required explicit `CUDA_VISIBLE_DEVICES=0`; leaving it unset skipped all 78 tests.
- With an empty TensorRT engine cache and default build settings, preparation of a CDA-VSR x2 fp16 engine for 960x540 input took about 19 seconds using the checkpoint supplied for v0.4.169. Sixty synthetic 960x540 inference frames then produced finite 1920x1080 output, with a median of 10.9 ms per frame.
- That historical check did not exercise real LMS input/output or NDI output on this GPU. The separate v0.4.396 functional check above does not turn these older engine timings into current performance results. The RTX A6000 results do not establish performance on the RTX 5090.

## Checkpoint requirements

The deb and archive contain no weights or engines. Receive a compatible checkpoint and `SHA256SUMS` as separate files from your provider and verify the received files as described in [INSTALL.md](INSTALL.md). Running an inference module requires a supported checkpoint, a GPU, and TensorRT.

CDA-VSR accepts a checkpoint trained with luma-range residuals and producer-exported safetensors metadata `cda_vsr.residual_unit=luma-range`. Checkpoints without this metadata are rejected at startup. Format conversion, extracting a weight set, or manually adding metadata does not establish compatible training or replace the producer's export.

The historical CDA-VSR x2 checkpoint associated with v0.4.165,
`cda-vsr-x2-otf79-it10000.safetensors`, had the following documented training scope. It is a different file from `cda-vsr-x2-ft_c6_p96_20261101_18k.safetensors`, used in the v0.4.398 installation example; the following conditions and quality figures must not be attributed to that newer file:

- The weights are schedule-free averaged weights from iteration 10,000, during
  training. They are not a final model.
- Low-resolution training input was 8-bit 4:2:0 H.265 encoded with x265, GOP 10,
  no B frames, and one reference frame. Training did not include streams with
  B frames, more references, longer GOPs, or other bit depths and chroma formats.
- The training project's evaluation on four materials (560 frames) reported
  mean PSNR 29.612 dB and mean SSIM 0.7539. streamflow has not evaluated image
  quality with this checkpoint.

SPANv2 requires a compatible SPANv2 safetensors checkpoint. Other network architectures, shapes, scales, and weights require validation by the corresponding inference module. See [ENGINE.md](ENGINE.md).

## Operational limits and unverified scope

- Image quality has not been evaluated. The v0.4.398 latency and audio observations above cover only the stated same-host CDA-VSR x2 fp16 runs; end-to-end latency for LMS output remains unmeasured. Other hosts, receivers, network paths, models, and checkpoints require separate validation.
- v0.4.398 is distributed for Linux x86_64 / Ubuntu 24.04 or later, including WSL2. The Windows batch files launch the Linux executable inside WSL2; they do not provide a native Windows executable. The historical measurements have their own validation scopes.
- SRT input and output, NDI input, and SDI are not registered in this application. Selecting `input.type: srt` or `output.type: srt` reports that SRT is unsupported and exits before the first frame.
- The standard native prefix includes LMS, NDI, CDA-VSR and SPANv2. LMS requires glibc 2.39 or later; NDI requires the host Avahi client library (`libavahi-client.so.3`). TensorRT is supplied by the host. Missing selected modules stop startup before the first frame with the component name and companion distribution.
- The core and output modules control send cadence, repetition of the last image with silence, audio sample count adjustment, and prefill waits. streamflow's `sent` statistic counts output send periods, including repeats; it does not establish arrival at a receiver.
- The CLI provides no operator command to exchange its input, output, inference model, or settings during operation. Stop and restart it to apply changes. Each process handles one stream; simultaneous connections to multiple inputs or outputs within a process are not supported.
- A second stop request can abort outstanding waits while work remains incomplete. Exit statistics describe observations, without guaranteeing completion or receiver delivery.
- Functional development tests use fakes at input/output SDK and inference engine boundaries. Those tests validate buffer ownership, settings, metadata, progress, and shutdown; they do not substitute for the real connection measurements above or evaluate real engine image quality.

Copyright © 2026 minamikik / sync.dev. All rights reserved.
