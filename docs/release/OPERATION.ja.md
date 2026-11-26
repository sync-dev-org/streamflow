# streamflow 操作マニュアル

対象 version は **0.4.398** です。streamflow は C++ の application で、一つの process が GPU 一枚で一つの stream を扱います。deb と archive は、LMS と NDI の入力と出力、CDA-VSR と SPANv2 の推論に使う標準の部品を持ちます。

## 導入

Ubuntu 24.04 以降の Linux x86_64 (WSL2 を含む) と、CUDA 13 に対応する NVIDIA の GPU 一枚と driver を用意します。WSL2 の driver は Windows の側に導入します。streamflow の実行に Python、uv、CUDA Toolkit は要りません。TensorRT は host の物を使います。

操作者は、提供者から deb または archive、使う model の checkpoint、`SHA256SUMS` を file として受け取ります。[INSTALL.md](INSTALL.md) の第 1 章に従い、受け取った配布物と checkpoint の両方を `SHA256SUMS` と照合します。

続いて、[INSTALL.md](INSTALL.md) の第 2 章に従い、host の Ubuntu の version に合う NVIDIA の CUDA apt repository を登録します。Ubuntu 24.04 は `ubuntu2404`、Ubuntu 26.04 は `ubuntu2604` の repository を使います。WSL2 でも同じで、`wsl-ubuntu` の repository は TensorRT を持たないので使いません。その後、受け取った deb を次の一回の command で導入します。

```bash
sudo apt install ./streamflow_0.4.398_amd64.deb
streamflow --help
```

apt は TensorRT の runtime (`libnvinfer11`、11.2 以上)、NDI が使う Avahi の client の library などを入れます。既定では `avahi-daemon` も入れます。mDNS による NDI の発見を使う時は、送り手側の daemon を起動します。

実行 file は `/opt/streamflow/bin/streamflow` にあり、`/usr/bin/streamflow` から呼べます。archive を使う場合は [INSTALL.md](INSTALL.md) の第 3 章に従って展開し、`prefix/bin/streamflow` を直接呼びます。archive と deb は同じ native の prefix を持ちます。

## 重みと設定

重みは deb と archive に入りません。操作者は、提供者から受け取った CDA-VSR ×2 の safetensors の checkpoint を、受け取った `SHA256SUMS` と照合して使います。SPANv2 には SPANv2 の checkpoint を使います。操作者は checkpoint の中身や metadata を書き換えません。互換性の条件は [ENGINE.md](ENGINE.md) を参照します。

操作者は、書込み可能な作業 directory に設定 file と重みを置きます。[settings-example.yaml](settings-example.yaml) をコピーし、LMS の接続先と channel、password、重みの path、入出力の寸法を実際の値に替えます。重みの相対 path は設定 file の directory から解決します。詳しい設定は [USAGE.md](USAGE.md) を参照します。

音声を出力する時は、操作者が `--output-audio-enabled` または `output.audio.enabled: true` を指定します。入力と推論は音声の到着を待たず、出力 module が送信の直前に画に対応する音声を確定します。音声を待つ時間の上限は `output.audio.max_wait_ms` で指定します。省略すると、streamflow は出力を開く時の frame 周期一つ分を ms で渡します (30000/1001 fps なら約 33.367 ms、25 fps なら 40 ms)。入力の周期が変わり出力を開き直す時は、新しい周期に合わせます。明示した値は周期が変わっても保ちます。0 は待たずに送る指定で、上限までに届かなかった音声は無音で埋めます。操作者は 0 以上の有限の数を指定します。負の数や数でない値を指定すると、streamflow は最初の frame の前に止まります。

NDI と LMS の出力では、操作者は `--output-delay-frames N` または `output.delay_frames: N` で遅延の枚数を選びます。既定は N = 2 です。core は入力 module が画を受け取った時刻から D = N × 一周期の後に画を選び、encode して送ります。保管の枚数 M は `--post-buffer-capacity-frames M` または `pipeline.post_buffer_capacity_frames: M` で選び、省略時は N に 1 を足した値になります。M は N 以上で、M − N の枠は早く届いた画を保つ余裕です。M を増やしても D は変わりません。M < N の時、streamflow は二つの項目と値を示して最初の frame の前に止まります。

処理が D に間に合う間、post buffer は画を捨てず、出力は前の画を繰り返しません。間に合わなかった画は音声と一緒に捨て、その周期は最後の画と無音を送ります。処理の時間を観測して D を変えることはありません。core は入力と出力の clock のずれだけを追い、位相の補正の上限には core の既定値を使います。

`output.prefill_frames` は NDI と LMS では 1 のみを受け付けます。不正な値を指定すると、streamflow は最初の frame の前に理由と `--output-delay-frames` を案内して止まります。

960×540 の入力を CDA-VSR ×2 で 1920×1080 にして NDI に送る例は次の形です。

```yaml
input:
  type: lms
  connection:
    id: "<LMS_ID>"
    password: "<LMS_PASSWORD>"
    channel: 1
inference:
  type: cda-vsr
  weights: models/cda-vsr-x2.safetensors
  input_width: 960
  input_height: 540
output:
  type: ndi
  width: 1920
  height: 1080
```

## LMS の入力の decode

操作者は `--input-option decode.library=nvdec` または設定 file の `input.decode.library: nvdec` で LMS の入力の decoder を選びます。値は文字列 `nvdec` と `ffmpeg` で、省略時は `nvdec` です。`ffmpeg` は受け取った access unit を受信の thread で software の decode します。`ffmpeg` では `input.decode_capacity_frames` は decode の前の queue の深さだけに効き、decoder の surface の数には当たりません。

`ffmpeg` は motion vector と残差を出せないため、CDA-VSR と組むと streamflow は理由を示して最初の frame の前に止まります。操作者は SPANv2 など motion vector と残差を要らない model を選びます。入力 module は走行中に decoder の library を切り替えません。操作者は設定を変える時に停止してから起動し直します。

## NDI の入力

操作者は `--input-type ndi` または設定 file の `input.type: ndi` で NDI の入力を選び、`source_name` に送り手が示す NDI の source の名前を指定します。名前は必須で、空文字と NUL を含む文字列は使えません。host の接頭辞、空白、記号を含め、実際の source 名をそのまま指定します。

操作者は `receive_delay_ms` に 0 から 2147483647 までの整数を ms 単位で指定できます。引数にも設定 file にも無い時は、streamflow が 120 ms を渡します。NDI SDK は映像と音声に同じ受信 delay を適用し、0 は追加の待ちを無くす指定です。入力 module の受信の側の待ちは、100 ms の遅延の目標に数えません。不正な名前や delay を指定すると、streamflow は項目と値を示し、最初の frame の前に終了 code 2 で止まります。

NDI の入力は motion vector と残差を出さないため、CDA-VSR と組めません。この組み合わせでは、streamflow が入力と推論の名前を示して最初の frame の処理前に止まります。操作者は SPANv2 と、それに対応する checkpoint を選びます。

```yaml
input:
  type: ndi
  source_name: "camera (Studio A)"
  receive_delay_ms: 120
inference:
  type: spanv2
  weights: models/spanv2-x2.safetensors
output:
  type: ndi
  source_name: streamflow
  width: 1920
  height: 1080
```

同じ構成を引数だけで指定する例は次の形です。操作者は source 名と重みの path を実際の値に替えます。

```bash
streamflow --input-type ndi --input-option "source_name='camera (Studio A)'" \
  --input-option receive_delay_ms=120 --inference-type spanv2 --inference-weights models/spanv2-x2.safetensors \
  --output-type ndi --output-option source_name=streamflow --output-width 1920 --output-height 1080
```

操作者が設定 file と引数の両方に同じ項目を指定した時は、streamflow が引数の値を使います。NDI の入力は届いた画を順に受け、frame sync を使いません。出力の周期の調整は、LMS の入力と同じく出力 module が行います。

## 起動と停止

deb から導入した場合、操作者は自分の作業 directory で次を実行します。root では実行しません。

```bash
streamflow --config settings.yaml
```

archive の場合は、展開した directory から `./prefix/bin/streamflow --config settings.yaml` を実行します。初回は重み、入力寸法、GPU に応じた TensorRT の engine を準備するため時間が掛かります。Python の環境を作る必要はありません。

操作者は設定 file を省いて、[USAGE.md](USAGE.md#startup-arguments) の引数だけでも起動できます。`--config` と引数を併せて渡す時は、引数の値を使います。`--inference-weights` の相対 path は起動時の作業 directory、設定 file の `inference.weights` の相対 path は設定 file の directory から読みます。

Windows からは、deb の `/opt/streamflow/share/streamflow/windows/` または archive の `prefix/share/streamflow/windows/` にある `streamflow-cdavsr.bat` か `streamflow-spanv2.bat` を Windows にコピーして使います。操作者は先頭の `WSL_DISTRO`、`STREAMFLOW_MODEL` (WSL の中の checkpoint の path)、`NDI_NAME` を変え、`streamflow-cdavsr.bat abc-defg 1 120` の形で LMS の ID、channel、receive_delay_ms を渡します。未指定の値は入力を求められ、receive_delay_ms の空欄は 120 ms になります。batch file は LMS → 選んだ model の ×2 → 1920×1080 の NDI と音声の引数を組み立て、WSL の `streamflow` を直接起動します。配置と準備は [INSTALL.md](INSTALL.md#windows-batch-files-for-wsl) にあります。

streamflow は、一行に一つの JSON で進捗、出来事、最後の統計を標準出力に出します。進捗は入力や engine の準備を待つ間も約 0.05 秒ごとに出ます。`sent` は出力 module の送信周期の数で、最後の画の繰り返しも数え、受信側への到達を表しません。最後の `summary` には thread P の処理時間 `frame_ms` (件数、平均、中央値、p90、最大、単位 ms) が出ます。この処理時間は、推論と変換、GPU の完了待ちを含み、送受信を通した遅延ではありません。統計の各項目は [USAGE.md](USAGE.md#reading-standard-output) を参照します。

起動時の問題は標準エラーに出します。部品が無い時は最初の frame の前に止まり、部品の名前と、core が返す配布名または prefix へ部品を足す案内を示します。部品が無い場合、読込みに失敗した場合、ABI が合わない場合を区別します。

停止する時は Ctrl+C を一回押すか、SIGTERM を一回送り、受け付けた処理と入出力の終了を待ちます。待っている間の二回目以降の SIGINT または SIGTERM は残りの待ちを打ち切り、`stop.aborted` と統計を出して終了 code 1 で終わります。打ち切る待ちが無い場合は通常の停止になります。CLI には走行中に入力、出力、model を交換する操作者向けの command は無いので、設定を変える時は停止してから起動し直します。

| 終了 code | 意味 |
|---|---|
| 0 | 通常の停止、起動時の接続待ちの停止、または `--help` |
| 1 | 運転中の異常、または停止の待ちの打切り |
| 2 | 引数、設定、部品、重みなどによる起動の失敗 |

起動の失敗で終了 code 2 になった場合、`summary` は出ません。操作者は標準エラーの項目と理由を読み、設定や実行環境を直して起動し直します。既知の制限と各 version の性能・遅延の計測条件は [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) を参照します。

Copyright © 2026 minamikik / sync.dev. All rights reserved.
