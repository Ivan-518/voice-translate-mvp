import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from server.audio import decode_base64, decode_browser_audio_to_pcm, wav_to_pcm_s16le
from server.config import get_settings
from server.models import AudioProcessRequest, BrowserAudioProcessRequest, HealthResponse, TextProcessRequest
from server.pipeline import create_default_pipeline

settings = get_settings()
if settings.hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
pipeline = create_default_pipeline(
    output_sample_rate=settings.output_sample_rate,
    asr_engine=settings.asr_engine,
    faster_whisper_model=settings.faster_whisper_model,
    faster_whisper_device=settings.faster_whisper_device,
    faster_whisper_compute_type=settings.faster_whisper_compute_type,
    faster_whisper_beam_size=settings.faster_whisper_beam_size,
    faster_whisper_vad_filter=settings.faster_whisper_vad_filter,
    faster_whisper_initial_prompt=settings.faster_whisper_initial_prompt,
    translation_engine=settings.translation_engine,
    nllb_model=settings.nllb_model,
    nllb_device=settings.nllb_device,
    nllb_max_new_tokens=settings.nllb_max_new_tokens,
    qwen_model=settings.qwen_model,
    qwen_device=settings.qwen_device,
    qwen_max_new_tokens=settings.qwen_max_new_tokens,
    llm_translation_base_url=settings.llm_translation_base_url,
    llm_translation_api_key=settings.llm_translation_api_key,
    llm_translation_model=settings.llm_translation_model,
    llm_translation_timeout=settings.llm_translation_timeout,
    tts_engine=settings.tts_engine,
    espeak_voice=settings.espeak_voice,
    espeak_speed=settings.espeak_speed,
)

app = FastAPI(title=settings.app_name)


@app.get("/")
async def index() -> dict:
    return {
        "app": settings.app_name,
        "status": "ok",
        "routes": {
            "health": "/health",
            "process_text": "/api/process-text",
            "process_audio": "/api/process-audio",
            "speech_websocket": "/ws/speech",
            "docs": "/docs",
        },
        "quick_test": "python -m client.record_once --seconds 3 --source-lang zh --target-lang en --play",
    }


@app.get("/home", response_class=HTMLResponse)
async def home() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice Translate MVP</title>
  <style>
    :root {
      color-scheme: light;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      background: #f5f7fb;
      color: #182033;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      padding: 28px 16px;
      background: #f5f7fb;
    }
    main {
      width: min(1080px, 100%);
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #dfe5ef;
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(24, 32, 51, 0.08);
      overflow: hidden;
    }
    header {
      padding: 22px 26px;
      border-bottom: 1px solid #e7ecf4;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }
    .status {
      font-size: 13px;
      color: #2f6f4e;
      background: #e9f7ef;
      border: 1px solid #cbe9d7;
      padding: 6px 10px;
      border-radius: 999px;
      white-space: nowrap;
    }
    section {
      padding: 22px 26px 26px;
      display: grid;
      gap: 16px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 7px;
      font-size: 13px;
      font-weight: 600;
      color: #4b5873;
    }
    select, button {
      font: inherit;
    }
    select {
      width: 100%;
      height: 40px;
      padding: 0 10px;
      border: 1px solid #cfd7e6;
      border-radius: 6px;
      background: #fff;
      color: #182033;
      outline: none;
    }
    select:focus {
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }
    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: #2563eb;
      color: white;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    button:disabled {
      background: #9aa7bb;
      cursor: wait;
    }
    .secondary {
      background: #edf2f7;
      color: #243047;
    }
    .danger {
      background: #dc2626;
    }
    .meters {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border: 1px solid #dfe5ef;
      border-radius: 8px;
      background: #fbfcfe;
    }
    .bar {
      height: 10px;
      background: #e5eaf2;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar span {
      display: block;
      width: 0%;
      height: 100%;
      background: #22c55e;
      transition: width 80ms linear;
    }
    .message {
      color: #5a6680;
      font-size: 13px;
      min-width: 160px;
      text-align: right;
    }
    .timeline {
      display: grid;
      gap: 10px;
    }
    .item {
      border: 1px solid #dfe5ef;
      border-radius: 8px;
      background: #ffffff;
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    .item-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
      color: #65708a;
    }
    .texts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .text-box {
      min-height: 74px;
      border: 1px solid #e2e7f0;
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .text-title {
      margin-bottom: 5px;
      font-size: 12px;
      font-weight: 700;
      color: #4b5873;
    }
    audio {
      width: 100%;
      height: 38px;
    }
    @media (max-width: 760px) {
      body { padding: 14px 10px; }
      header, section { padding-left: 16px; padding-right: 16px; }
      header { align-items: flex-start; flex-direction: column; }
      .controls { grid-template-columns: 1fr 1fr; }
      .texts { grid-template-columns: 1fr; }
      .meters { grid-template-columns: 1fr; }
      .message { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Voice Translate MVP</h1>
      <div class="status" id="status">服务检测中</div>
    </header>
    <section>
      <div class="controls">
        <label>源语言
          <select id="sourceLang">
            <option value="zh">中文</option>
            <option value="en">English</option>
            <option value="auto">自动检测</option>
          </select>
        </label>
        <label>目标语言
          <select id="targetLang">
            <option value="en">English</option>
            <option value="zh">中文</option>
            <option value="ja">日本語</option>
            <option value="ko">한국어</option>
          </select>
        </label>
        <button id="startBtn">开始同传</button>
        <button class="danger" id="stopBtn" disabled>停止同传</button>
      </div>
      <div class="meters">
        <div class="bar"><span id="levelBar"></span></div>
        <span class="message" id="message">等待开始</span>
      </div>
      <div class="timeline" id="timeline"></div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const messageEl = document.getElementById("message");
    const levelBar = document.getElementById("levelBar");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const timeline = document.getElementById("timeline");

    let audioContext = null;
    let sourceNode = null;
    let processorNode = null;
    let mediaStream = null;
    let running = false;
    let speaking = false;
    let segmentChunks = [];
    let segmentMs = 0;
    let silenceMs = 0;
    let segmentId = 0;
    let playQueue = [];
    let playing = false;

    const startThreshold = 0.025;
    const stopThreshold = 0.014;
    const silenceLimitMs = 850;
    const minSegmentMs = 550;
    const maxSegmentMs = 5600;

    async function checkHealth() {
      try {
        const res = await fetch("/health");
        if (!res.ok) throw new Error("health failed");
        statusEl.textContent = "服务在线";
      } catch {
        statusEl.textContent = "服务异常";
      }
    }

    function getLangPayload() {
      return {
        source_lang: document.getElementById("sourceLang").value,
        target_lang: document.getElementById("targetLang").value,
        voice_id: "default"
      };
    }

    startBtn.addEventListener("click", startInterpreting);
    stopBtn.addEventListener("click", stopInterpreting);

    async function startInterpreting() {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new AudioContext();
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        processorNode.onaudioprocess = onAudioProcess;
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        running = true;
        speaking = false;
        segmentChunks = [];
        segmentMs = 0;
        silenceMs = 0;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        messageEl.textContent = "监听中";
      } catch (err) {
        messageEl.textContent = "无法访问麦克风：" + err.message;
      }
    }

    function stopInterpreting() {
      running = false;
      if (speaking) finalizeSegment();
      if (processorNode) processorNode.disconnect();
      if (sourceNode) sourceNode.disconnect();
      if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
      if (audioContext) audioContext.close();
      processorNode = null;
      sourceNode = null;
      mediaStream = null;
      audioContext = null;
      speaking = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      levelBar.style.width = "0%";
      messageEl.textContent = "已停止";
    }

    function onAudioProcess(event) {
      if (!running) return;
      const input = event.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(input);
      const rms = calcRms(chunk);
      const chunkMs = chunk.length / audioContext.sampleRate * 1000;
      levelBar.style.width = Math.min(100, Math.round(rms * 900)) + "%";

      if (!speaking && rms >= startThreshold) {
        speaking = true;
        segmentChunks = [];
        segmentMs = 0;
        silenceMs = 0;
        messageEl.textContent = "说话中";
      }

      if (!speaking) return;

      segmentChunks.push(chunk);
      segmentMs += chunkMs;
      if (rms < stopThreshold) {
        silenceMs += chunkMs;
      } else {
        silenceMs = 0;
      }

      if ((segmentMs >= minSegmentMs && silenceMs >= silenceLimitMs) || segmentMs >= maxSegmentMs) {
        finalizeSegment();
      }
    }

    function calcRms(samples) {
      let sum = 0;
      for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
      return Math.sqrt(sum / Math.max(1, samples.length));
    }

    function finalizeSegment() {
      if (!speaking || segmentMs < minSegmentMs) {
        resetSegment();
        return;
      }
      const chunks = segmentChunks.slice();
      const sampleRate = audioContext.sampleRate;
      const duration = segmentMs;
      const id = ++segmentId;
      resetSegment();
      messageEl.textContent = "处理中";
      submitSegment(id, chunks, sampleRate, duration);
    }

    function resetSegment() {
      speaking = false;
      segmentChunks = [];
      segmentMs = 0;
      silenceMs = 0;
      if (running) messageEl.textContent = "监听中";
    }

    async function submitSegment(id, chunks, sampleRate, duration) {
      const row = createTimelineItem(id, duration);
      try {
        const wavBytes = encodeWav(chunks, sampleRate);
        const res = await fetch("/api/process-browser-audio", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_base64: uint8ToBase64(wavBytes),
            audio_format: "wav",
            ...getLangPayload()
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        updateTimelineItem(row, data);
        if (!data.engine_trace.includes("no-speech")) {
          enqueueAudio(data.audio_base64);
        }
      } catch (err) {
        row.querySelector(".state").textContent = "失败";
        row.querySelector(".translated").textContent = err.message;
      }
    }

    function createTimelineItem(id, duration) {
      const item = document.createElement("div");
      item.className = "item";
      item.innerHTML = `
        <div class="item-head">
          <span>#${id} · ${(duration / 1000).toFixed(1)}s</span>
          <span class="state">处理中</span>
        </div>
        <div class="texts">
          <div class="text-box"><div class="text-title">原文</div><div class="source">-</div></div>
          <div class="text-box"><div class="text-title">译文</div><div class="translated">-</div></div>
        </div>
        <audio controls></audio>`;
      timeline.prepend(item);
      return item;
    }

    function updateTimelineItem(item, data) {
      item.querySelector(".state").textContent = data.engine_trace.join(" -> ");
      item.querySelector(".source").textContent = data.source_text || "-";
      item.querySelector(".translated").textContent = data.engine_trace.includes("no-speech")
        ? "未识别到有效语音"
        : (data.translated_text || "-");
      if (data.timings_ms) {
        const timingText = Object.entries(data.timings_ms)
          .map(([key, value]) => `${key}:${value}ms`)
          .join(" · ");
        item.querySelector(".state").textContent += " · " + timingText;
      }
      item.querySelector("audio").src = URL.createObjectURL(base64ToBlob(data.audio_base64, "audio/wav"));
    }

    function enqueueAudio(base64) {
      playQueue.push(base64);
      playNext();
    }

    function playNext() {
      if (playing || playQueue.length === 0) return;
      playing = true;
      const audio = new Audio(URL.createObjectURL(base64ToBlob(playQueue.shift(), "audio/wav")));
      audio.onended = () => {
        playing = false;
        playNext();
      };
      audio.onerror = () => {
        playing = false;
        playNext();
      };
      audio.play().catch(() => {
        playing = false;
        messageEl.textContent = "浏览器阻止自动播放";
      });
    }

    function base64ToBlob(base64, type) {
      const bin = atob(base64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
      return new Blob([bytes], { type });
    }

    function uint8ToBase64(bytes) {
      let binary = "";
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, chunk);
      }
      return btoa(binary);
    }

    function encodeWav(chunks, sampleRate) {
      const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
      const buffer = new ArrayBuffer(44 + length * 2);
      const view = new DataView(buffer);
      writeString(view, 0, "RIFF");
      view.setUint32(4, 36 + length * 2, true);
      writeString(view, 8, "WAVE");
      writeString(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, "data");
      view.setUint32(40, length * 2, true);

      let offset = 44;
      for (const chunk of chunks) {
        for (let i = 0; i < chunk.length; i += 1) {
          const sample = Math.max(-1, Math.min(1, chunk[i]));
          view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
          offset += 2;
        }
      }
      return new Uint8Array(buffer);
    }

    function writeString(view, offset, value) {
      for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
    }

    checkHealth();
  </script>
</body>
</html>
"""


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@app.post("/api/process-text")
async def process_text(request: TextProcessRequest):
    return await pipeline.process_text(
        text=request.text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        voice_id=request.voice_id,
    )


@app.post("/api/process-audio")
async def process_audio(request: AudioProcessRequest):
    audio = decode_base64(request.audio_base64)
    return await pipeline.process_audio(
        audio=audio,
        sample_rate=request.sample_rate,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        voice_id=request.voice_id,
    )


@app.post("/api/process-browser-audio")
async def process_browser_audio(request: BrowserAudioProcessRequest):
    browser_audio = decode_base64(request.audio_base64)
    if request.audio_format == "wav":
        pcm_audio, sample_rate = wav_to_pcm_s16le(browser_audio)
    else:
        sample_rate = 16000
        pcm_audio = decode_browser_audio_to_pcm(browser_audio, input_format=request.audio_format, sample_rate=sample_rate)
    return await pipeline.process_audio(
        audio=pcm_audio,
        sample_rate=sample_rate,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        voice_id=request.voice_id,
    )


@app.websocket("/ws/speech")
async def speech_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_config = {
        "source_lang": settings.default_source_lang,
        "target_lang": settings.default_target_lang,
        "voice_id": "default",
    }

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "config":
                session_config.update({key: value for key, value in message.items() if key in session_config})
                await websocket.send_json({"type": "config_ack", "config": session_config})
                continue

            if message_type != "audio_chunk":
                await websocket.send_json({"type": "error", "message": f"unsupported message type: {message_type}"})
                continue

            audio = decode_base64(message["audio_base64"])
            result = await pipeline.process_audio(
                audio=audio,
                sample_rate=int(message.get("sample_rate", 16000)),
                source_lang=session_config["source_lang"],
                target_lang=session_config["target_lang"],
                voice_id=session_config["voice_id"],
            )
            await websocket.send_json(
                {
                    "type": "result",
                    "chunk_id": message.get("chunk_id"),
                    "payload": result.model_dump(),
                }
            )
    except WebSocketDisconnect:
        return
