# streamflow

streamflow is a real-time video application. It receives a live video stream, upscales it on an NVIDIA GPU, and sends the result to an output at a steady cadence. When processing falls behind or the input pauses, streamflow drops the oldest frames and repeats the last image with silence, so the output keeps running.

- Input and output: LMS input and output, NDI output
- Super-resolution: CDA-VSR and SPANv2, running on TensorRT
- One process handles one stream. streamflow is a C++ application and runs without a Python environment.

## Introduction

The [introduction page](https://sync-dev-org.github.io/streamflow/) explains how streamflow works, with diagrams of the pipeline, frame dropping, output timing, audio and timecode, and components.

