# 架构说明

## MVP 边界

第一版只解决一件事：证明“本地音频输入 -> 服务器处理 -> 本地拿到目标语言语音”这条链路可运行。

| 模块 | 放置位置 | 当前实现 | 后续替换 |
| --- | --- | --- | --- |
| 音频采集 | 本地 | `client.record_once` 单次录音 | 持续采集、VAD、设备选择 |
| 音频上传 | 本地 -> 服务器 | WebSocket JSON + base64 PCM | 二进制帧、Opus、WebRTC |
| ASR | 服务器 | `StubAsrEngine`，可配置 `FasterWhisperAsrEngine` | FunASR / 流式 ASR |
| 翻译 | 服务器 | `StubTranslationEngine`，可配置 `GoogleTranslationEngine` | LLM API / NLLB / M2M100 |
| TTS | 服务器 | `StubTtsEngine`，可配置 `Pyttsx3TtsEngine` | Edge TTS / Kokoro / CosyVoice |
| 音频后处理 | 服务器 | `PassthroughPostProcessor` | RVC / 音量归一化 / 降噪 |
| 音频输出 | 本地 | 保存 WAV 文件，可选播放到指定设备 | 播放队列、VB-CABLE、Voicemeeter |

## 核心接口

### HTTP 文本链路

```text
POST /api/process-text
```

用于调试翻译和 TTS，不依赖麦克风。

### HTTP 音频链路

```text
POST /api/process-audio
```

用于调试单个音频片段。

### WebSocket 音频链路

```text
WS /ws/speech
```

当前每个 `audio_chunk` 都会触发一次完整处理。后续可以改成：

```text
audio_chunk
audio_chunk
audio_chunk
utterance_end -> process
```

## 后续接 RVC

RVC 不应该侵入 ASR、翻译、TTS 主流程。推荐只作为 TTS 后处理：

```text
translated_text
  -> TTS
  -> RvcPostProcessor
  -> converted_audio
```

代码上新增一个后处理器即可：

```python
class RvcPostProcessor(AudioPostProcessor):
    name = "rvc-postprocessor"

    async def process(self, audio: bytes, sample_rate: int, audio_format: str) -> TtsResult:
        converted_audio = await rvc_service.convert(audio, sample_rate)
        return TtsResult(audio=converted_audio, sample_rate=sample_rate, audio_format=audio_format)
```

然后在 `create_default_pipeline()` 中替换：

```python
postprocessors=[RvcPostProcessor(...)]
```

## 延迟优化路线

| 阶段 | 优化点 |
| --- | --- |
| 1 | 本地 VAD，减少无效音频上传 |
| 2 | WebSocket 改二进制帧，减少 base64 开销 |
| 3 | ASR 支持流式或短窗口增量识别 |
| 4 | 翻译按短句/半句处理 |
| 5 | TTS 支持流式播放 |
| 6 | RVC 支持 chunk 级推理和播放缓存 |
