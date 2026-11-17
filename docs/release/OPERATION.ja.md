# streamflow 操作マニュアル

対象 version は **0.4.398** です。streamflow は C++ の application で、一つの process が GPU 一枚で一つの stream を扱います。deb と archive は、LMS の入力と出力、NDI の出力、CDA-VSR と SPANv2 の推論に使う標準の部品を持ちます。

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

音声を出力する時は、操作者が `--audio` または `output.audio.enabled: true` を指定します。入力と推論は音声の到着を待たず、出力 module が送信の直前に画に対応する音声を確定します。音声を待つ時間の上限は `output.audio.max_wait_ms` で指定し、省略時は入力の frame 周期一回分 (30000/1001 fps では約 33.4 ms) です。0 は待たずに送る指定で、上限までに届かなかった音声は無音で埋めます。操作者は 0 以上の有限の数を指定します。負の数や数でない値を指定すると、streamflow は最初の frame の前に止まります。

`input.audio_wait_frame_limit` と `--input-option audio_wait_frame_limit=...` は受け付けません。操作者は以前の設定からこの項目を除き、必要なら出力の `output.audio.max_wait_ms` を ms で指定します。streamflow は古い項目を受け取ると、移行先を示して最初の frame の前に止まります。

NDI と LMS の出力では、送信の時刻を post buffer の待ちに追従させる機能が既定で有効です。目標の待ちは 4 ms、一周期あたりの補正の上限は 0.1 ms です。操作者は `output.follow_input.enabled: false`、または `--output-option follow_input.enabled=false` で追従を無効にできます。

960×540 の入力を CDA-VSR ×2 で 1920×1080 にして NDI に送る例は次の形です。

```yaml
input:
  type: lms
  connection:
    id: "<LMS_ID>"
    password: "<LMS_PASSWORD>"
    channel: 1
inference:
  arch: cda-vsr
  model: models/cda-vsr-x2.safetensors
  input_width: 960
  input_height: 540
output:
  type: ndi
  width: 1920
  height: 1080
```

## 起動と停止

deb から導入した場合、操作者は自分の作業 directory で次を実行します。root では実行しません。

```bash
streamflow --config settings.yaml
```

archive の場合は、展開した directory から `./prefix/bin/streamflow --config settings.yaml` を実行します。初回は重み、入力寸法、GPU に応じた TensorRT の engine を準備するため時間が掛かります。Python の環境を作る必要はありません。

操作者は設定 file を省いて、[USAGE.md](USAGE.md#startup-arguments) の引数だけでも起動できます。`--config` と引数を併せて渡す時は、引数の値を使います。`--model` の相対 path は起動時の作業 directory、設定 file の `inference.model` の相対 path は設定 file の directory から読みます。

Windows からは、deb の `/opt/streamflow/share/streamflow/windows/` または archive の `prefix/share/streamflow/windows/` にある `streamflow-cdavsr.bat` か `streamflow-spanv2.bat` を Windows にコピーして使います。操作者は先頭の `WSL_DISTRO`、`STREAMFLOW_MODEL` (WSL の中の checkpoint の path)、`NDI_NAME` を変え、`streamflow-cdavsr.bat abc-defg 1 100` の形で LMS の ID、channel、delay_ms を渡します。未指定の値は入力を求められ、delay_ms の空欄は 100 ms になります。batch file は LMS → 選んだ model の ×2 → 1920×1080 の NDI と音声の引数を組み立て、WSL の `streamflow` を直接起動します。配置と準備は [INSTALL.md](INSTALL.md#windows-batch-files-for-wsl) にあります。

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
