# streamflow インストールマニュアル

対象バージョン: **0.4.398** / Linux x86_64・Ubuntu・WSL2

Windows x86_64 の zip / EXE installer を使う場合は [Windows への導入](#native-windows)を参照してください。

受け取った配布ファイルの確認から、LMS の入力を CDA-VSR ×2 で処理して NDI へ出力する最初の起動までを説明します。操作は Ubuntu のターミナルで行います。Windows のコマンドプロンプトで行う操作は第 6 章に示します。

## 1. 対象環境と受け取るファイル

| 項目 | 条件 |
|---|---|
| OS | Ubuntu 24.04 以降の x86_64。WSL2 上の Ubuntu を含みます |
| C ランタイム | glibc 2.39 以降 |
| GPU | NVIDIA GPU 1 枚。配布する部品の対象は sm 86・89・120 |
| ドライバー | CUDA 13 に対応する NVIDIA ドライバー |
| TensorRT | `libnvinfer11` 11.2 以上。NVIDIA の CUDA apt リポジトリから導入します |
| ネットワーク | LMS の入力元と NDI の受信側に到達できること。apt の導入時はインターネット接続も必要です |

通常は deb を選びます。任意のディレクトリへ展開して使う場合は archive を選びます。どちらも実行ファイルと LMS、NDI、CDA-VSR、SPANv2 の標準部品を持ちます。streamflow の実行に Python、uv、CUDA Toolkit、nvcc の導入は必要ありません。

| 受け取るファイル | 用途 |
|---|---|
| `streamflow_0.4.398_amd64.deb` | apt で導入する場合に使います |
| `streamflow-0.4.398-linux_x86_64.tar.gz` | 展開して使う場合に使います |
| `cda-vsr-x2-ft_c6_p96_20261101_18k.safetensors` | CDA-VSR ×2 の checkpoint。deb / archive とは別のファイルです |
| `SHA256SUMS` | 受け取ったファイルの SHA-256 を照合します |

### 導入前にファイルを照合する

選んだ deb または archive と、checkpoint、`SHA256SUMS` を同じディレクトリに置き、そのディレクトリで実行します。

```bash
sha256sum --check --ignore-missing SHA256SUMS
```

選んだ配布ファイルと checkpoint の両方が `OK` と表示されることを確認します。`--ignore-missing` は、選ばなかった方の配布ファイルを照合対象から外すための指定です。必要な checkpoint まで欠けていないか、ファイル名と結果を確認してください。`FAILED` が出た場合は導入を進めず、該当ファイルを受け取り直します。

## 2. GPU と TensorRT の準備

### NVIDIA ドライバーを用意する

Ubuntu を直接動かす場合は、GPU に合った CUDA 13 対応の NVIDIA ドライバーを用意します。WSL2 の場合は Windows 側に NVIDIA ドライバーを導入し、WSL の Ubuntu 内には Linux 用 NVIDIA ドライバーを導入しません。[NVIDIA のドライバー配布ページ](https://www.nvidia.com/drivers/)と [CUDA on WSL の公式手順](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)を参照してください。

WSL2 を使う場合は、Windows のコマンドプロンプトで `wsl.exe --list --verbose` を実行し、使用する Ubuntu の `VERSION` が `2` であることを確認します。以降の準備と導入は、その Ubuntu のターミナルで行います。

### NVIDIA の CUDA apt リポジトリを登録する

Ubuntu のバージョンと CPU の種類を確認し、対応するリポジトリ名を選びます。

```bash
cat /etc/os-release
uname -m
```

| Ubuntu | `REPO` に指定する値 |
|---|---|
| 24.04 | `ubuntu2404` |
| 26.04 | `ubuntu2604` |

次は Ubuntu 24.04 の例です。Ubuntu 26.04 では最初の `REPO` の値を `ubuntu2604` に替えます。ほかのバージョンは、[NVIDIA の Ubuntu 向け公式手順](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)で対応するリポジトリを確認してください。

```bash
sudo apt update
sudo apt install curl ca-certificates
REPO=ubuntu2404
curl -fsSLO "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
apt-cache policy libnvinfer11
```

`libnvinfer11` の候補が 11.2 以上であることを確認します。WSL2 でも Ubuntu のバージョンに対応するリポジトリを使います。`wsl-ubuntu` リポジトリからは TensorRT を導入できません。`cuda-keyring` は apt の登録先を設定し、別の NVIDIA リポジトリ用の `cuda-keyring` が入っている場合は置き換えます。

この配布物の build に使った TensorRT SDK は **11.2.1.2** です。TensorRT は配布物に含まれません。deb は次章の apt 操作でランタイムを導入します。archive では次章の依存パッケージも導入します。

## 3. streamflow を導入する

### deb を使う場合

受け取った deb のあるディレクトリで実行します。

```bash
sudo apt install ./streamflow_0.4.398_amd64.deb
streamflow --help
```

apt は `libnvinfer11 >= 11.2`、`libavahi-client3`、`libstdc++6`、`libgcc-s1`、`zlib1g`、`libc6 >= 2.39` を依存として導入します。既定では推奨パッケージの `avahi-daemon` も導入します。NDI の mDNS による発見を使う場合は、streamflow を動かす Ubuntu 側で daemon を起動します。

```bash
sudo systemctl enable --now avahi-daemon
systemctl is-active avahi-daemon
```

systemd を使わない WSL2 では、代わりに `sudo service avahi-daemon start` と `service avahi-daemon status` を使います。NDI の受信側が送信元の IP を直接指定し、mDNS による発見を必要としない運用では、apt の `--no-install-recommends` で daemon の導入を省けます。

| 配置 | 内容 |
|---|---|
| `/opt/streamflow` | 実行ファイルと標準部品 |
| `/usr/bin/streamflow` | `/opt/streamflow/bin/streamflow` を指すコマンド |
| `/usr/share/doc/streamflow` | 導入手順、設定例、利用者向け文書 |

checkpoint は同梱されません。設定と checkpoint は操作者が書き込めるディレクトリに置きます。通常の起動は `sudo` を付けずに行います。

### archive を使う場合

操作者が書き込める場所で展開します。第 2 章の apt リポジトリの登録後、依存パッケージも導入します。

```bash
tar -xzf streamflow-0.4.398-linux_x86_64.tar.gz
cd streamflow-0.4.398-linux_x86_64
sha256sum --check SHA256SUMS
sudo apt install libnvinfer11 libavahi-client3 libstdc++6 libgcc-s1 zlib1g avahi-daemon
./prefix/bin/streamflow --help
```

ここで照合する `SHA256SUMS` は archive 内部のファイル用です。第 1 章の配布ファイルと checkpoint 用の `SHA256SUMS` とは別です。NDI の mDNS を使う場合は deb と同様に Avahi を起動します。

archive はディレクトリ全体をまとめて移動できます。実行ファイルだけを取り出さず、`prefix/` の配置を保ちます。以下の起動例の `streamflow` は、展開先の `prefix/bin/streamflow` の絶対パスに替えます。archive 内の追加の学習用ツールは別の環境で使う物で、streamflow の起動には必要ありません。詳細は同梱の `TRAINING.md` を参照してください。

## 4. checkpoint を置き、引数で最初に起動する

### 操作者の作業ディレクトリを作る

第 1 章で照合した checkpoint のあるディレクトリで実行します。

```bash
mkdir -p "$HOME/streamflow/models"
cp cda-vsr-x2-ft_c6_p96_20261101_18k.safetensors \
  "$HOME/streamflow/models/"
cd "$HOME/streamflow"
```

checkpoint の中身は変更せずに使います。TensorRT engine のパスを指定する操作は不要です。SPANv2 を選ぶ場合は、対応する checkpoint を別途用意します。詳細は [ENGINE.md](ENGINE.md) を参照してください。

### LMS → CDA-VSR ×2 → NDI の起動例

接続 ID `abc-defg` と channel `1` を実際の接続に替えます。この例は password が空の接続です。password がある場合は `connection.password` の空文字列を実際の値に替えてください。

```bash
streamflow \
  --input-type lms \
  --input-option "connection.id='abc-defg'" \
  --input-option "connection.password=''" \
  --input-option connection.channel=1 \
  --input-option receive_delay_ms=120 \
  --inference-type cda-vsr \
  --inference-weights "$HOME/streamflow/models/cda-vsr-x2-ft_c6_p96_20261101_18k.safetensors" \
  --inference-option precision=fp16 \
  --inference-option input_width=960 \
  --inference-option input_height=540 \
  --output-type ndi --output-option "source_name='streamflow'" \
  --output-width 1920 --output-height 1080 --output-audio-enabled
```

入力 960 × 540、出力 1920 × 1080 は設定例です。入力の幅と高さは実際の映像に、出力の幅と高さは出力したい大きさに合わせます。`receive_delay_ms=120` は LMS の受信側が待つ時間を ms で指定し、省略した場合も 120 ms です。

`--output-audio-enabled` は出力の音声を有効にし、省略すると音声を送りません。CLI の `--inference-weights` に相対パスを書く場合は、実行したディレクトリが基点です。初回は GPU を使って不足する TensorRT engine を準備するため時間がかかります。

出力 module は送信の直前に音声を確定します。待ちの上限を省略すると、streamflow は出力を開く時の frame 周期一つ分を ms で渡します (30000/1001 fps なら約 33.367 ms)。入力の周期が変わり出力を開き直す時は、新しい周期に合わせます。変更する場合は `--output-option audio.max_wait_ms=12.5` のように ms で指定します。0 は待たずに送り、届いていない音声を無音で埋めます。NDI と LMS の出力では、出口が各画を、入力 module がその画を受け取った時刻から D = N × 一周期の後に選んで送ります。遅延の枚数 N の既定は 2、post buffer の保管の枚数 M の既定は N に 3 を足した値です。詳しい設定は [USAGE.md](USAGE.md#output) を参照してください。

## 5. YAML の設定ファイルで起動する

第 4 章の `$HOME/streamflow` に、次の内容を `settings.yaml` として保存します。接続と映像の幅・高さは実際の値に替えます。

```yaml
input:
  type: lms
  connection:
    id: "abc-defg"
    password: ""
    channel: 1
  receive_delay_ms: 120
inference:
  type: cda-vsr
  weights: models/cda-vsr-x2-ft_c6_p96_20261101_18k.safetensors
  precision: fp16
  input_width: 960
  input_height: 540
output:
  type: ndi
  source_name: streamflow
  width: 1920
  height: 1080
  audio:
    enabled: true
```

```bash
streamflow --config "$HOME/streamflow/settings.yaml"
```

YAML の `inference.weights` の相対パスは、設定ファイルのあるディレクトリが基点です。この例では `$HOME/streamflow/models/` の checkpoint を読みます。YAML 内はこの例の相対パスか、実際の絶対パスを使ってください。

引数と設定ファイルを併用すると、同じ項目は引数が優先されます。たとえば `streamflow --config settings.yaml --output-width 1280 --output-height 720` は出力の大きさだけを替えます。入力・出力・推論の種類と、部品に必要な項目は引数または設定ファイルで指定します。全項目は [USAGE.md](USAGE.md)、設定例は [settings-example.yaml](settings-example.yaml) を参照してください。

音声待ち上限の YAML の項目は `output.audio.max_wait_ms` です。操作者は 0 以上の有限の数を指定します。

### 起動後の確認と停止

NDI の受信側で、出力名 `streamflow` の映像と音声を確認します。NDI の出力は最初の入力 frame が届いてから開きます。入力前は NDI の出力が見つからない場合があります。入力が途切れると再開を待ち、出力に渡せる新しい画が無い時は最後の画と無音を送ります。

標準出力には進捗・動作状況と終了時の統計が JSON の行で出ます。起動時のエラーは標準エラー出力を確認します。停止は `Ctrl+C` を一回押し、処理と統計の出力を待ちます。待っている間にもう一回停止要求を送ると、残っている待ちを打ち切り、終了コードが 0 以外になります。

<a id="windows-batch-files-for-wsl"></a>

## 6. Windows から WSL2 の streamflow を起動する

配布物の `streamflow-cdavsr.bat` は CDA-VSR、`streamflow-spanv2.bat` は SPANv2 を選んで起動します。deb での配置は `/opt/streamflow/share/streamflow/windows/`、archive は `prefix/share/streamflow/windows/` です。

使うファイルを Windows 側のフォルダーにコピーし、テキストエディターで先頭の設定を編集します。

```bat
set "WSL_DISTRO=Ubuntu-24.04"
set "STREAMFLOW_MODEL=/home/<WSL_USER>/streamflow/models/cda-vsr-x2-ft_c6_p96_20261101_18k.safetensors"
set "NDI_NAME=streamflow-cdavsr"
```

`WSL_DISTRO` は `wsl.exe --list --verbose` に表示された名前に合わせます。`<WSL_USER>` は実際の Ubuntu のユーザー名に替えます。`STREAMFLOW_MODEL` は Windows のパスではなく、**WSL 内の checkpoint の絶対パス**です。SPANv2 の場合は SPANv2 の checkpoint を指定します。deb は batch file の初期設定に書かれた重みのパスを作らないため、必ず変更します。

最初に Windows のコマンドプロンプトで確認します。

```bat
wsl.exe -d Ubuntu-24.04 -- streamflow --help
```

続いて batch file を置いたフォルダーで、使用するモデルの一方を実行します。`abc-defg` と `1` は実際の LMS の ID と channel に替えます。

```bat
streamflow-cdavsr.bat abc-defg 1 120
```

SPANv2 の場合は `streamflow-spanv2.bat abc-defg 1 120` です。引数は左から LMS の ID、channel、受信の待ち時間 (ms) です。省略した値は画面で入力を求められ、待ち時間を空欄にすると 120 ms を使います。

batch file は入力 960 × 540・fp16・出力 1920 × 1080・音声有効の LMS → ×2 → NDI を起動し、password は空文字列です。password や入力サイズを変える場合は、第 4 章の引数を使うか、batch file 内の対応する引数を編集します。

archive を使う場合は、batch file 内の `-- streamflow` を展開先の `prefix/bin/streamflow` の WSL 内の絶対パスに替える方法があります。第 6 章の `wsl.exe ... -- streamflow --help` も同じ絶対パスに替えて確認します。deb では `/usr/bin/streamflow` を使えるため変更は不要です。

## 7. 導入の確認と削除

### 導入した内容を確認する

deb のバージョンと実行ファイルを確認します。

```bash
dpkg-query -W -f='${Package} ${Version}\n' streamflow
readlink -f /usr/bin/streamflow
streamflow --help
cat /opt/streamflow/share/streamflow/release.json
```

バージョンは `0.4.398`、実行ファイルは `/opt/streamflow/bin/streamflow` です。archive は展開先で次を実行します。

```bash
./prefix/bin/streamflow --help
cat prefix/share/streamflow/release.json
```

`release.json` は配布物と TensorRT SDK のバージョンを記録します。ライブラリの license と出どころは `share/streamflow/packages/` にあり、deb は `/opt/streamflow/`、archive は `prefix/` からの相対パスです。

`--help` の成功はコマンドが動くことの確認です。GPU での推論、LMS の接続、NDI の受信は第 4 章または第 5 章の起動で別途確認します。運転の説明は [OPERATION.ja.md](OPERATION.ja.md)、既知の制限は [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) を参照してください。

### 削除する

streamflow を停止した後、deb は次で削除します。

```bash
sudo apt remove streamflow
```

`sudo apt purge streamflow` も使用できます。操作者のディレクトリの設定と checkpoint は別に保管します。TensorRT や Avahi をほかのアプリケーションも使う場合は削除しないでください。

archive は停止後に設定と checkpoint を退避し、展開した `streamflow-0.4.398-linux_x86_64` ディレクトリをファイルマネージャーなどで削除します。Windows の batch file の実行パスも不要なら削除・変更します。

engine cache は `$XDG_CACHE_HOME/tensoracle`、未設定なら `~/.cache/tensoracle` にあります。ほかのアプリケーションからも使われる場合があるため、streamflow と一緒に削除する必要はありません。checkpoint、入力サイズ、GPU、TensorRT などが変わると、次の起動で engine を作り直す場合があります。

## 8. よくある失敗と対処

| 症状 | 確認と対処 |
|---|---|
| checksum が `FAILED` | 導入を止め、該当ファイルと同じ配布分の `SHA256SUMS` を受け取り直します。ファイル名の変更前に照合します |
| `libnvinfer11` の候補が無い | 第 2 章の Ubuntu のバージョン、NVIDIA の apt 登録と `sudo apt update` の結果を確認します。WSL2 でも Ubuntu 用を使います |
| `libnvinfer.so.11` を読み込めない | `dpkg-query -W libnvinfer11` で 11.2 以上の導入を確認します。archive は第 3 章の依存も導入します |
| GPU を初期化できない | CUDA 13 対応の NVIDIA ドライバーを確認します。WSL2 は Windows 側のドライバーと、対象の Ubuntu が WSL2 であることを確認します |
| `streamflow` が見つからない | deb は `/usr/bin/streamflow` を確認します。archive は `prefix/bin/streamflow` の絶対パスで起動します |
| checkpoint が見つからない | 別ファイルの checkpoint を置いたか確認します。CLI の相対パスは実行ディレクトリ、YAML は設定ファイルのディレクトリが基点です |
| モデルの読込み・engine の準備に失敗 | モデルと checkpoint の組合せ、読取り権限と cache の書込み権限を確認し、表示された診断に従います |
| 初回の出力まで時間がかかる | engine の準備中か標準出力を確認します。初回の準備は GPU と時間が必要です。入力サイズは実際の映像に合わせます |
| NDI の出力が見つからない | 最初の LMS frame、出力名、Avahi の起動を確認します。送信元と受信側の到達性、WSL2 と Windows のネットワーク・ファイアウォール設定も確認します |
| 音声が出ない | `--output-audio-enabled` または `output.audio.enabled: true` と入力の音声を確認します。出力する新しい画が無い時は無音を送ります |
| 設定や部品のエラー | 診断された項目と部品を [USAGE.md](USAGE.md) と照合し、設定を直して起動し直します |

問い合わせる際は、配布物のバージョン、Ubuntu / WSL2 の別、起動コマンドまたは設定、表示された診断を添えてください。

<a id="native-windows"></a>

## 9. Windows への導入

Windows x86_64 用の zip と installer は、同じ実行ファイルと標準部品 (LMS、NDI、CDA-VSR、SPANv2) を持ちます。streamflow の実行に Python、Visual Studio、CUDA Toolkit、NSIS の導入は必要ありません。設定と checkpoint は操作者が書き込めるフォルダーに別途置きます。

### host の準備

| 項目 | 導入する物 |
|---|---|
| NVIDIA ドライバー | GPU に対応する CUDA 13 対応の Windows 用ドライバー。CUDA 13 系の互換性の下限は R580 系です。使用する TensorRT の条件も確認してください |
| TensorRT | この build が使用する SDK は **11.2.1.2**。Windows x64 / CUDA 13 用の runtime を導入し、`nvinfer_11.dll` のある `lib` フォルダーを PATH に追加します |
| MSVC Redistributable | Microsoft Visual C++ v14 **x64**。build tools (MSVC 19.44 / toolset 14.44) 以上の版を使います。Microsoft が提供する最新の対応版を導入してください |
| Desktop Video | DeckLink を使う構成で必要になる Blackmagic の機器用ドライバーです。この配布物の標準部品には SDI module を含めていません |

NVIDIA ドライバーは [NVIDIA の配布ページ](https://www.nvidia.com/drivers/)から GPU に合う物を選びます。[CUDA の driver 互換性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)も確認してください。MSVC Redistributable は [Microsoft の手順と x64 の配布先](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)から導入します。MSVC の C++ runtime DLL は streamflow の配布物に含めません。

利用者は [NVIDIA の Windows 用 zip の導入手順](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-zip.html)に従い、CUDA 13 用の TensorRT の zip を展開します。`nvinfer_11.dll` 一つだけをコピーせず、engine の準備に使う builder resource なども含めて配置します。Windows の「環境変数」で、その DLL を持つフォルダーを PATH に追加し、新しいターミナルを開きます。全 user で使う場合は、各 user がそのフォルダーを読み取れ、各 user の PATH から見つかるようにします。

```bat
where nvinfer_11.dll
```

streamflow と installer は、PATH の順に要求する DLL を探し、最初に見つかったフォルダーを使います。TensorRT 10 系だけでは起動も installer の導入もできません。配布物が要求する DLL 名と SDK の version は `share\streamflow\windows-runtime.json`、配布物の version と source commit は `share\streamflow\release.json` に記録します。

installer は導入する管理者の account の Machine と User の環境変数から PATH を組み立てます。UAC で別の管理者の account を使う場合は、その account の User PATH を検査します。コマンドプロンプトでその process だけに追加した PATH は使いません。全 user で使う時は、全 user が読み取れる TensorRT のフォルダーを Machine の PATH に置くことを推奨します。

NVIDIA ドライバー、TensorRT、Desktop Video、MSVC Redistributable は host の前提です。zip と installer はそれらを導入・削除しません。Windows の確認環境は Windows 11 Pro for Workstations 10.0.22631、Visual Studio 2022 17.14 / MSVC 19.44、CUDA 13.2、TensorRT 11.2.1.2、NDI SDK 6.3.2 です。Windows の version の下限をこの確認環境から定めるものではありません。

### 配布ファイルを確認する

受け取った `streamflow-X.Y.Z-windows_x86_64.zip` または `.exe` と `SHA256SUMS` を同じフォルダーに置きます。PowerShell で SHA-256 を計算し、`SHA256SUMS` に書かれた同じファイル名の値と比較します。

```powershell
Get-FileHash .\streamflow-X.Y.Z-windows_x86_64.exe -Algorithm SHA256
Get-Content .\SHA256SUMS
```

値が違う場合は導入を進めず、同じ配布分のファイルを受け取り直します。checkpoint は配布物と別のファイルであり、その checksum も確認します。

### EXE の installer を使う

installer はその PC の全 user に導入し、管理者の権限を求めます。既定の導入先は `C:\Program Files\streamflow` です。installer は導入先の `bin` を system の PATH に追加し、`bin\streamflow.exe` の受信を許す Windows Defender Firewall の規則を全 profile に追加します。

管理者の権限を持たない user は zip を使ってください。zip 自体の展開には管理者の権限は要りません。受信を許す firewall の規則は、管理者に追加を依頼してください。

installer に code signing は行っていません。SmartScreen などが発行元を確認できないと表示する場合があります。受け取った配布元と checksum を確認して操作してください。

TensorRT の DLL が PATH から見つからない場合、installer は入手先を表示して導入を止めます。host の準備を済ませ、新しいターミナルまたは Explorer から再実行してください。

コマンドから導入する場合は、管理者のターミナルで実行します。

```bat
start "" /wait streamflow-X.Y.Z-windows_x86_64.exe /S
echo %ERRORLEVEL%
```

`/S` は画面を出さない導入です。終了 code は成功が `0`、失敗が `0` 以外です。GUI と `/S` は同じ TensorRT の検査を行い、DLL が無い時はその名前と入手先を `%TEMP%\streamflow-setup.log` (UTF-16LE) に記録します。GUI では同じ理由を画面にも表示します。

導入後は新しいターミナルを開き、`streamflow --help` を実行します。同じ導入先の更新と同じ version の再導入に対応します。導入済みの version より古い version は拒否します。MSI で導入した streamflow が残っている場合、EXE は変更を始める前に止まります。Windows の「インストールされているアプリ」から MSI で導入した streamflow を削除してから EXE を実行してください。

削除は Windows の「インストールされているアプリ」から streamflow を選ぶか、導入先の `uninstall.exe` を開きます。削除にも管理者の権限が必要です。installer は自分が導入したファイル、自分が足した system の PATH の項目と firewall の規則を削除します。別に置いた設定と checkpoint は残ります。導入と削除の処理が失敗した時は、file、PATH、firewall とアプリの登録を処理前の状態へ戻します。

画面を出さずに削除し、完了と終了 code を取得する場合は、管理者の PowerShell で実行します。`_?=` の導入先は引数の最後に置き、導入先を変えた場合は実際の path に替えてください。

```powershell
Copy-Item 'C:\Program Files\streamflow\uninstall.exe' "$env:TEMP\streamflow-uninstall.exe"
$result = Start-Process "$env:TEMP\streamflow-uninstall.exe" -ArgumentList '/S _?=C:\Program Files\streamflow' -Wait -PassThru
$result.ExitCode
```

削除の理由と結果は `%TEMP%\streamflow-uninstall.log` (UTF-16LE) に記録します。

### zip を使う

操作者が書き込める新しいフォルダーに zip 全体を展開します。PowerShell の例です。

```powershell
Expand-Archive .\streamflow-X.Y.Z-windows_x86_64.zip -DestinationPath .\streamflow-unpacked
& .\streamflow-unpacked\streamflow-X.Y.Z-windows_x86_64\bin\streamflow.exe --help
```

zip は PATH を変更しません。実行ファイルの絶対パスを使うか、必要に応じて展開先の `bin` を自分の PATH に追加します。移動する時は展開したフォルダー全体を移動し、`bin` や DLL だけを取り出さないでください。削除する時は streamflow を停止し、設定と checkpoint を退避してから展開先を削除します。手動で追加した PATH も取り除きます。

zip は firewall の規則を追加しません。LMS などの入力を受信するには、次の手順で規則を追加してください。

### firewall の規則を手動で追加する

管理者として PowerShell を開き、`$program` に実際の `streamflow.exe` の絶対 path を指定します。次の例は installer の既定の導入先です。zip を使う場合は展開先の `bin\streamflow.exe` に置き換えます。

```powershell
$program = (Resolve-Path 'C:\Program Files\streamflow\bin\streamflow.exe').Path
New-NetFirewallRule -Name 'streamflow-manual-inbound' -DisplayName 'streamflow manual inbound' `
  -Direction Inbound -Action Allow -Program $program -Profile Any
```

この規則は指定した実行ファイルの受信だけを許可します。すべての program に対して port を開く規則ではありません。installer が追加する規則の表示名は `streamflow inbound` です。installer が規則を追加済みの場合、手動で規則を重ねる必要はありません。

手動で追加した規則は installer の uninstall や zip の削除では消えません。配布物を削除する時は、管理者の PowerShell で次の command を実行します。zip の展開先を移動した時は、古い規則を削除し、新しい絶対 path で追加し直してください。

```powershell
Remove-NetFirewallRule -Name 'streamflow-manual-inbound'
```

### 起動と NDI の通知

[設定例](settings-example.yaml)を操作者のフォルダーに置き、接続値と checkpoint の path を編集して `streamflow --config settings.yaml` で起動します。YAML の model の相対 path は設定ファイルからの相対です。Windows の絶対 path には `C:/Users/.../model.safetensors` のように slash を使えます。`--help` の成功は起動の確認です。GPU の推論や接続の確認は別に行います。[利用方法](USAGE.md)と[既知の制限](KNOWN_LIMITATIONS.md)も参照してください。

streamflow は [NDI](https://ndi.video/) の技術を使用します。NDI® is a registered trademark of Vizrt NDI AB. NDI SDK 6.3.2 の Windows の runtime とともに、`pixtreme\transport\ndi\_ndi\Processing.NDI.Lib.Licenses.txt` を同梱します。NDI の利用条件と第三者の通知はこのファイルで確認できます。各部品の license と出どころは `share\streamflow\packages\` にあります。

Copyright © 2026 minamikik / sync.dev. All rights reserved.
