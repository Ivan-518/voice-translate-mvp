# Voice Translate MVP

一个“本地采集 + 服务器 ASR/翻译/TTS + 本地播放/虚拟麦克风输出”的同声传译 MVP 骨架。

当前版本先不接 RVC，先把主链路拆清楚并跑通协议：

```text
本地客户端
  -> 麦克风采集 / 音频切片
  -> WebSocket / HTTP 上传

服务器
  -> ASR
  -> 翻译
  -> TTS
  -> 音频后处理钩子

本地客户端
  -> 播放音频
  -> 后续输出到 VB-CABLE / Voicemeeter
```

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `server/main.py` | FastAPI 入口，提供健康检查、文本处理、音频处理和 WebSocket |
| `server/pipeline.py` | ASR -> 翻译 -> TTS -> 后处理主流程 |
| `server/engines.py` | 可替换引擎接口和默认 stub 实现 |
| `server/audio.py` | PCM/WAV 编解码、小音频生成工具 |
| `server/models.py` | API 请求和响应模型 |
| `server/config.py` | 环境变量配置 |
| `client/text_demo.py` | 不用麦克风的文本到语音接口调试脚本 |
| `client/record_once.py` | 本地录音一次并通过 WebSocket 发给服务器 |
| `client/list_devices.py` | 列出本地音频输入/输出设备，方便选择虚拟声卡 |
| `docs/architecture.md` | 架构、接口和后续接 RVC 的设计说明 |

## 快速启动

```powershell
cd E:\PythonProject\voice-translate-mvp
pip install -e .
uvicorn server.main:app --reload --host 127.0.0.1 --port 8010
```

也可以直接用 Python 入口启动：

```powershell
python run_server.py
```

需要热重载时：

```powershell
python run_server.py --reload
```

打开健康检查：

```text
http://127.0.0.1:8010/health
```

## 文本链路测试

不需要麦克风，不需要 GPU：

```powershell
python -m client.text_demo --text "我马上到" --target-lang en
```

脚本会调用服务端 `/api/process-text`，并把返回的 stub TTS 音频保存到：

```text
outputs/text_demo.wav
```

安装 `.[client]` 后可以直接播放：

```powershell
python -m client.text_demo --text "我马上到" --target-lang en --play
```

## 单次录音测试

需要安装客户端音频依赖：

```powershell
pip install -e ".[client]"
python -m client.record_once --seconds 3 --target-lang en
```

当前 ASR 是 stub，因此它不会真的识别语音内容，只会返回一段可调试文本。后续把 `server/engines.py` 中的 `StubAsrEngine` 替换为 faster-whisper/FunASR 即可。

## 启用 faster-whisper ASR

安装 ASR、翻译、TTS 和客户端依赖：

```powershell
pip install -e ".[asr,translate,tts,client]"
```

AutoDL / 国内服务器建议直接使用 HuggingFace 镜像：

```text
HF_ENDPOINT=https://hf-mirror.com
```

复制配置文件：

```powershell
copy .env.example .env
```

把 `.env` 中的 ASR 改为：

```text
ASR_ENGINE=faster_whisper
TRANSLATION_ENGINE=google
TTS_ENGINE=edge
FASTER_WHISPER_MODEL=small
FASTER_WHISPER_DEVICE=auto
FASTER_WHISPER_COMPUTE_TYPE=default
EDGE_TTS_VOICE=en-US-JennyNeural
EDGE_TTS_RATE=+0%
EDGE_TTS_VOLUME=+0%
```

低配机器可以先用：

```text
FASTER_WHISPER_MODEL=tiny
FASTER_WHISPER_COMPUTE_TYPE=int8
```

中文识别不准时，优先把模型调大：

```text
FASTER_WHISPER_MODEL=base
FASTER_WHISPER_COMPUTE_TYPE=int8
FASTER_WHISPER_BEAM_SIZE=5
FASTER_WHISPER_VAD_FILTER=false
FASTER_WHISPER_INITIAL_PROMPT=以下是普通话中文语音，请准确转写为简体中文。
```

浏览器端已经做了 VAD 自动切句，因此这里建议先关闭 faster-whisper 内置 VAD，避免短中文片段被二次过滤。

如果要使用百度短语音识别，把 ASR 切到百度：

```text
ASR_ENGINE=baidu
BAIDU_ASR_API_KEY=你的百度智能云语音 API Key
BAIDU_ASR_SECRET_KEY=你的百度智能云语音 Secret Key
BAIDU_ASR_CUID=voice-translate-mvp
BAIDU_ASR_DEV_PID=1537
BAIDU_ASR_ENDPOINT=https://vop.baidu.com/server_api
BAIDU_ASR_TOKEN_URL=https://aip.baidubce.com/oauth/2.0/token
BAIDU_ASR_TIMEOUT=15
BAIDU_ASR_SAMPLE_RATE=16000
BAIDU_ASR_MIN_INTERVAL=2.0
```

其中 `1537` 适合普通话短语音识别。注意百度 ASR 的 `API Key / Secret Key` 来自百度智能云语音识别，不是百度翻译的 APPID/密钥。
`BAIDU_ASR_MIN_INTERVAL` 用来限制百度 ASR 请求频率，避免浏览器 VAD 切得太碎时触发 `3305 request pv too much`。触发间隔保护或百度返回 `3305` 时，服务端会把当前片段标记为 `no-speech` 并跳过，不会自动切换到其他 ASR 引擎。

如果有 GPU，可以进一步用：

```text
FASTER_WHISPER_MODEL=small
FASTER_WHISPER_DEVICE=cuda
FASTER_WHISPER_COMPUTE_TYPE=float16
```

识别准确率和延迟的取舍：

| 配置 | 准确率 | 延迟 | 适用场景 |
| --- | --- | --- | --- |
| `tiny + int8` | 低 | 低 | 验证链路 |
| `base + int8` | 中 | 中 | CPU 中文 MVP |
| `small + float16` | 较高 | 中高 | GPU 同传 |
| `medium + float16` | 高 | 高 | 离线或高质量模式 |

然后重启服务：

```powershell
uvicorn server.main:app --reload --host 127.0.0.1 --port 8010
```

再录音测试：

```powershell
python -m client.record_once --seconds 3 --source-lang zh --target-lang en --play
```

此时链路是：

```text
麦克风录音
  -> faster-whisper 识别
  -> Google Translate 或本地 NLLB 翻译
  -> Edge TTS / espeak / pyttsx3 TTS
  -> 本地播放或输出到指定音频设备
```

AutoDL 访问不了 `translate.google.com` 时，改用本地 NLLB 翻译：

```text
TRANSLATION_ENGINE=nllb
NLLB_MODEL=facebook/nllb-200-distilled-600M
NLLB_DEVICE=cuda
NLLB_MAX_NEW_TOKENS=128
```

Linux 服务器不建议用 `pyttsx3`。优先用 Edge TTS，声音比 espeak 自然：

```text
TTS_ENGINE=edge
EDGE_TTS_VOICE=en-US-JennyNeural
EDGE_TTS_RATE=+0%
EDGE_TTS_VOLUME=+0%
```

如果 Edge TTS 网络不可用，或只需要离线验证链路，再改用命令行 `espeak-ng`：

```bash
apt-get update
apt-get install -y espeak-ng
```

`.env` 中设置：

```text
TTS_ENGINE=espeak
ESPEAK_VOICE=en-us
ESPEAK_SPEED=165
```

检查 espeak 是否能生成有效 WAV：

```bash
espeak-ng -v en-us -w /tmp/espeak-test.wav "hello"
ls -lh /tmp/espeak-test.wav
file /tmp/espeak-test.wav
```

如果文件大小为 0 或不是 WAV，重新安装：

```bash
apt-get update
apt-get install -y --reinstall espeak-ng espeak-ng-data
```

服务器首次下载 NLLB 模型时建议使用 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python tools/download_models.py --translation
```

也可以一次性下载 ASR 和 NLLB：

```bash
python tools/download_models.py --all
```

如果 NLLB 对短中文口语翻译不准，推荐改用本地 Qwen 翻译：

```text
TRANSLATION_ENGINE=qwen
QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct
QWEN_DEVICE=cuda
QWEN_MAX_NEW_TOKENS=128
```

预下载 Qwen：

```bash
python -m pip install -e ".[asr,qwen-translate,tts]"
python tools/download_models.py --qwen
```

32GB 显存可以进一步尝试更高质量的 7B：

```text
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
```

如果要使用 OpenAI-compatible 大模型 API 翻译，例如 Packy API：

```text
TRANSLATION_ENGINE=llm
LLM_TRANSLATION_BASE_URL=https://www.packyapi.com/v1
LLM_TRANSLATION_API_KEY=你的 API Key
LLM_TRANSLATION_MODEL=gpt-4o-mini
LLM_TRANSLATION_TIMEOUT=30
LLM_TRANSLATION_TEMPERATURE=
```

如果要使用百度通用翻译 API：

```text
TRANSLATION_ENGINE=baidu
BAIDU_TRANSLATE_APP_ID=你的 APP ID
BAIDU_TRANSLATE_SECRET_KEY=你的密钥
BAIDU_TRANSLATE_ENDPOINT=https://fanyi-api.baidu.com/api/trans/vip/translate
BAIDU_TRANSLATE_TIMEOUT=10
```

如果要输出到虚拟麦克风，先查看设备名：

```powershell
python -m client.list_devices
```

然后指定输出设备：

```powershell
python -m client.record_once --seconds 3 --source-lang zh --target-lang en --play --output-device "CABLE Input"
```

查看可用音频设备：

```powershell
python -m client.list_devices
```

如果已经安装 VB-CABLE 或 Voicemeeter，可以把设备名传给 `--output-device`：

```powershell
python -m client.record_once --seconds 3 --play --output-device "CABLE Input"
```

## AutoDL 推荐部署命令

```bash
cd ~/autodl-tmp/voice-translate-mvp
git pull origin main
python -m pip install -e ".[asr,local-translate,tts]"
apt-get update
apt-get install -y espeak-ng
```

`.env` 推荐配置：

```text
HF_ENDPOINT=https://hf-mirror.com
ASR_ENGINE=faster_whisper
TRANSLATION_ENGINE=baidu
TTS_ENGINE=edge
SOURCE_LANG=zh
TARGET_LANG=en
OUTPUT_SAMPLE_RATE=24000
FASTER_WHISPER_MODEL=base
FASTER_WHISPER_DEVICE=cuda
FASTER_WHISPER_COMPUTE_TYPE=float16
FASTER_WHISPER_BEAM_SIZE=5
FASTER_WHISPER_VAD_FILTER=false
FASTER_WHISPER_INITIAL_PROMPT=以下是普通话中文语音，请准确转写为简体中文。

# 如果使用百度 ASR，改成 ASR_ENGINE=baidu 并填写：
# BAIDU_ASR_API_KEY=你的百度智能云语音 API Key
# BAIDU_ASR_SECRET_KEY=你的百度智能云语音 Secret Key
# BAIDU_ASR_CUID=voice-translate-mvp
# BAIDU_ASR_DEV_PID=1537
# BAIDU_ASR_SAMPLE_RATE=16000
# BAIDU_ASR_MIN_INTERVAL=2.0
BAIDU_TRANSLATE_APP_ID=你的 APP ID
BAIDU_TRANSLATE_SECRET_KEY=你的密钥
BAIDU_TRANSLATE_ENDPOINT=https://fanyi-api.baidu.com/api/trans/vip/translate
BAIDU_TRANSLATE_TIMEOUT=10
EDGE_TTS_VOICE=en-US-JennyNeural
EDGE_TTS_RATE=+0%
EDGE_TTS_VOLUME=+0%
```

预下载模型：

```bash
python tools/download_models.py --asr
```

启动服务：

```bash
python run_server.py --host 0.0.0.0 --port 6006
```

## 后续接入顺序

| 阶段 | 目标 | 修改位置 |
| --- | --- | --- |
| 1 | 接真实 ASR | 已支持 `FasterWhisperAsrEngine` 和 `BaiduAsrEngine` |
| 2 | 接真实翻译 | 已支持 Google、NLLB、Qwen、OpenAI-compatible API 和百度翻译引擎 |
| 3 | 接真实 TTS | 已支持 Edge TTS、espeak 和 pyttsx3，推荐通过 `TTS_ENGINE=edge` 启用 |
| 4 | 本地虚拟麦克风 | `client/` 新增输出设备选择和持续播放队列 |
| 5 | 接 RVC | 在 `AudioPostProcessor` 后处理钩子里增加 `RvcPostProcessor` |

## 和 Whispering Tiger 的借鉴点

| Whispering Tiger 思路 | 本项目对应设计 |
| --- | --- |
| ASR、翻译、TTS 分层 | `SpeechPipeline` 组合多个引擎 |
| 插件后处理 | `AudioPostProcessor`，未来接 RVC |
| WebSocket 实时状态 | `/ws/speech` |
| Profile 配置 | 后续扩展到客户端配置文件 |
| 本机音频设备管理 | 放在 `client/`，服务端不直接碰本地声卡 |
