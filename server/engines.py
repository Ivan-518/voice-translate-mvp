import asyncio
import base64
import hashlib
import io
import json
import secrets
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from server.audio import (
    generate_silence_wav,
    generate_tone_wav,
    pcm_s16le_duration_seconds,
    resample_pcm_s16le,
)


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    diagnostic: str = ""


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source_lang: str
    target_lang: str


@dataclass(frozen=True)
class TtsResult:
    audio: bytes
    sample_rate: int
    audio_format: str = "wav"


class AsrEngine:
    name = "asr"

    async def transcribe(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
        raise NotImplementedError


class TranslationEngine:
    name = "translation"

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        raise NotImplementedError


class TtsEngine:
    name = "tts"

    async def synthesize(self, text: str, target_lang: str, voice_id: str) -> TtsResult:
        raise NotImplementedError


class AudioPostProcessor:
    name = "postprocessor"

    async def process(self, audio: bytes, sample_rate: int, audio_format: str) -> TtsResult:
        raise NotImplementedError


class StubAsrEngine(AsrEngine):
    name = "stub-asr"

    async def transcribe(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
        duration = pcm_s16le_duration_seconds(audio, sample_rate)
        lang = "zh" if source_lang == "auto" else source_lang
        return AsrResult(text=f"这是一段约 {duration:.1f} 秒的测试语音", language=lang)


class FasterWhisperAsrEngine(AsrEngine):
    name = "faster-whisper-asr"

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        self.model = None

    def _get_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 faster-whisper。请先运行：pip install -e \".[asr]\""
            ) from exc

        if self.model is None:
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self.model

    async def transcribe(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
        return await asyncio.to_thread(self._transcribe_sync, audio, sample_rate, source_lang)

    def _transcribe_sync(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
        wav_bytes = self._pcm_to_wav_buffer(audio, sample_rate)
        language = None if source_lang == "auto" else source_lang
        model = self._get_model()
        segments, info = model.transcribe(
            wav_bytes,
            language=language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt or None,
            condition_on_previous_text=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        detected_language = source_lang if source_lang != "auto" else (info.language or "unknown")
        return AsrResult(text=text, language=detected_language)

    @staticmethod
    def _pcm_to_wav_buffer(audio: bytes, sample_rate: int) -> io.BytesIO:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio)
        buffer.seek(0)
        return buffer


class BaiduAsrEngine(AsrEngine):
    name = "baidu-asr"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        cuid: str = "voice-translate-mvp",
        dev_pid: int = 1537,
        endpoint: str = "https://vop.baidu.com/server_api",
        token_url: str = "https://aip.baidubce.com/oauth/2.0/token",
        timeout: float = 15.0,
        sample_rate: int = 16000,
        min_interval: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.cuid = cuid
        self.dev_pid = dev_pid
        self.endpoint = endpoint
        self.token_url = token_url
        self.timeout = timeout
        self.sample_rate = sample_rate
        self.min_interval = min_interval
        self._access_token: str | None = None
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()

    async def transcribe(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
        return await asyncio.to_thread(self._transcribe_sync, audio, sample_rate, source_lang)

    def _transcribe_sync(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
        if not audio:
            return AsrResult(text="", language=self._result_language(source_lang), diagnostic="baidu-asr-empty-audio")
        if not self.api_key or not self.secret_key:
            raise RuntimeError("BAIDU_ASR_API_KEY 或 BAIDU_ASR_SECRET_KEY 未配置")

        if self._should_skip_for_rate_limit():
            return AsrResult(text="", language=self._result_language(source_lang), diagnostic="baidu-asr-rate-limited-local")

        pcm_audio = resample_pcm_s16le(audio, sample_rate, self.sample_rate)
        payload = {
            "format": "pcm",
            "rate": self.sample_rate,
            "channel": 1,
            "cuid": self.cuid,
            "token": self._get_access_token(),
            "dev_pid": self.dev_pid,
            "speech": base64.b64encode(pcm_audio).decode("ascii"),
            "len": len(pcm_audio),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度 ASR 接口 HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"百度 ASR 接口不可用：{exc.reason}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"百度 ASR 响应不是 JSON：{body[:500]}") from exc

        if data.get("err_no") != 0:
            if data.get("err_no") == 3305:
                return AsrResult(text="", language=self._result_language(source_lang), diagnostic="baidu-asr-rate-limited-3305")
            raise RuntimeError(f"百度 ASR 错误 {data.get('err_no')}: {data.get('err_msg')}")
        text = "".join(data.get("result") or []).strip()
        diagnostic = "baidu-asr-empty-result" if not text else ""
        return AsrResult(text=text, language=self._result_language(source_lang), diagnostic=diagnostic)

    def _should_skip_for_rate_limit(self) -> bool:
        if self.min_interval <= 0:
            return False

        now = time.monotonic()
        with self._request_lock:
            if now - self._last_request_at < self.min_interval:
                return True
            self._last_request_at = now
            return False

    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token

        params = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            }
        )
        url = f"{self.token_url}?{params}"
        request = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度 ASR token HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"百度 ASR token 接口不可用：{exc.reason}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"百度 ASR token 响应不是 JSON：{body[:500]}") from exc
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"百度 ASR token 响应缺少 access_token：{data}")
        self._access_token = token
        return token

    @staticmethod
    def _result_language(source_lang: str) -> str:
        return "zh" if source_lang in {"", "auto", "unknown"} else source_lang


class StubTranslationEngine(TranslationEngine):
    name = "stub-translation"

    _dictionary = {
        "我马上到": "I'll be there soon.",
        "你好": "Hello.",
        "谢谢": "Thank you.",
        "这是一段测试语音": "This is a test voice message.",
    }

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        translated = self._dictionary.get(text.strip())
        if translated is None:
            translated = f"[{target_lang}] {text}"
        return TranslationResult(text=translated, source_lang=source_lang, target_lang=target_lang)


class GoogleTranslationEngine(TranslationEngine):
    name = "google-translation"

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return await asyncio.to_thread(self._translate_sync, text, source_lang, target_lang)

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise RuntimeError(
                "未安装 deep-translator。请先运行：pip install -e \".[translate]\""
            ) from exc

        source = self._normalize_lang(source_lang, is_source=True)
        target = self._normalize_lang(target_lang, is_source=False)
        try:
            translated = GoogleTranslator(source=source, target=target).translate(text)
        except Exception as exc:
            raise RuntimeError(f"Google 翻译不可用：{exc}") from exc
        return TranslationResult(text=translated, source_lang=source_lang, target_lang=target_lang)

    @staticmethod
    def _normalize_lang(lang: str, is_source: bool) -> str:
        if is_source and lang in {"", "auto", "unknown"}:
            return "auto"
        return {
            "zh": "zh-CN",
            "zh_cn": "zh-CN",
            "zh-CN": "zh-CN",
            "en": "en",
            "ja": "ja",
            "jp": "ja",
            "ko": "ko",
            "fr": "fr",
            "de": "de",
            "es": "es",
            "ru": "ru",
        }.get(lang, lang)


class NllbTranslationEngine(TranslationEngine):
    name = "nllb-translation"

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 128) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.torch_device = None

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return await asyncio.to_thread(self._translate_sync, text, source_lang, target_lang)

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text="", source_lang=source_lang, target_lang=target_lang)

        tokenizer, model, torch_device = self._get_model()
        src_code = self._to_nllb_code(source_lang, is_source=True)
        tgt_code = self._to_nllb_code(target_lang, is_source=False)

        tokenizer.src_lang = src_code
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(torch_device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return TranslationResult(text=translated, source_lang=source_lang, target_lang=target_lang)

    def _get_model(self):
        if self.model is not None and self.tokenizer is not None and self.torch_device is not None:
            return self.tokenizer, self.model, self.torch_device

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "未安装本地翻译依赖。请先运行：pip install -e \".[local-translate]\""
            ) from exc

        if self.device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = self.device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(torch_device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.torch_device = torch_device
        return tokenizer, model, torch_device

    @staticmethod
    def _to_nllb_code(lang: str, is_source: bool) -> str:
        if is_source and lang in {"", "auto", "unknown"}:
            return "zho_Hans"
        return {
            "zh": "zho_Hans",
            "zh-cn": "zho_Hans",
            "zh_cn": "zho_Hans",
            "zh-CN": "zho_Hans",
            "en": "eng_Latn",
            "ja": "jpn_Jpan",
            "jp": "jpn_Jpan",
            "ko": "kor_Hang",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "es": "spa_Latn",
            "ru": "rus_Cyrl",
        }.get(lang, "eng_Latn" if not is_source else "zho_Hans")


class QwenTranslationEngine(TranslationEngine):
    name = "qwen-translation"

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 128) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.torch_device = None

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return await asyncio.to_thread(self._translate_sync, text, source_lang, target_lang)

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text="", source_lang=source_lang, target_lang=target_lang)

        tokenizer, model, torch_device = self._get_model()
        prompt = self._build_prompt(text, source_lang, target_lang)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional real-time interpreter. "
                    "Translate faithfully and concisely. "
                    "Return only the translated sentence, with no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        encoded_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([encoded_text], return_tensors="pt").to(torch_device)
        generated = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        output_ids = generated[0][inputs.input_ids.shape[-1] :]
        translated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        translated = self._clean_output(translated)
        return TranslationResult(text=translated, source_lang=source_lang, target_lang=target_lang)

    def _get_model(self):
        if self.model is not None and self.tokenizer is not None and self.torch_device is not None:
            return self.tokenizer, self.model, self.torch_device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 Qwen 翻译依赖。请先运行：pip install -e \".[qwen-translate]\""
            ) from exc

        if self.device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = self.device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=torch_device if torch_device == "cuda" else None,
            trust_remote_code=True,
        )
        if torch_device != "cuda":
            model.to(torch_device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.torch_device = torch_device
        return tokenizer, model, torch_device

    @staticmethod
    def _build_prompt(text: str, source_lang: str, target_lang: str) -> str:
        source_name = QwenTranslationEngine._lang_name(source_lang, default="Chinese")
        target_name = QwenTranslationEngine._lang_name(target_lang, default="English")
        return (
            f"Translate the following {source_name} text into {target_name}. "
            "Keep names, brands, numbers, and punctuation faithful. "
            "Do not add facts. Do not summarize. Text:\n"
            f"{text}"
        )

    @staticmethod
    def _lang_name(lang: str, default: str) -> str:
        return {
            "auto": default,
            "unknown": default,
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh_cn": "Chinese",
            "zh-CN": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "jp": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
        }.get(lang, default)

    @staticmethod
    def _clean_output(text: str) -> str:
        cleaned = text.strip()
        for prefix in ("Translation:", "Translated text:", "译文：", "翻译："):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip("\"'“”")


class OpenAICompatibleTranslationEngine(TranslationEngine):
    name = "llm-translation"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        temperature: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return await asyncio.to_thread(self._translate_sync, text, source_lang, target_lang)

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text="", source_lang=source_lang, target_lang=target_lang)
        if not self.api_key:
            raise RuntimeError("LLM_TRANSLATION_API_KEY 未配置")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional real-time interpreter. "
                        "Translate faithfully and concisely. "
                        "Return only the translated sentence. "
                        "Do not add explanations, prefixes, bullet points, or facts."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(text, source_lang, target_lang),
                },
            ],
        }
        if self.temperature.strip():
            payload["temperature"] = float(self.temperature)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"LLM 翻译响应不是 JSON：{body[:500]}") from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM 翻译接口 HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM 翻译接口不可用：{exc.reason}") from exc

        try:
            translated = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM 翻译响应格式异常：{data}") from exc

        return TranslationResult(
            text=self._clean_output(translated),
            source_lang=source_lang,
            target_lang=target_lang,
        )

    @staticmethod
    def _build_prompt(text: str, source_lang: str, target_lang: str) -> str:
        source_name = QwenTranslationEngine._lang_name(source_lang, default="Chinese")
        target_name = QwenTranslationEngine._lang_name(target_lang, default="English")
        return (
            f"Translate from {source_name} to {target_name}. "
            "Keep brand names, numbers, and punctuation faithful. "
            "Text:\n"
            f"{text}"
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        cleaned = text.strip()
        for prefix in ("Translation:", "Translated text:", "译文：", "翻译："):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip("\"'“”")


class BaiduTranslationEngine(TranslationEngine):
    name = "baidu-translation"

    def __init__(
        self,
        app_id: str,
        secret_key: str,
        endpoint: str = "https://fanyi-api.baidu.com/api/trans/vip/translate",
        timeout: float = 10.0,
    ) -> None:
        self.app_id = app_id
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.timeout = timeout

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return await asyncio.to_thread(self._translate_sync, text, source_lang, target_lang)

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(text="", source_lang=source_lang, target_lang=target_lang)
        if not self.app_id or not self.secret_key:
            raise RuntimeError("BAIDU_TRANSLATE_APP_ID 或 BAIDU_TRANSLATE_SECRET_KEY 未配置")

        source = self._normalize_lang(source_lang, is_source=True)
        target = self._normalize_lang(target_lang, is_source=False)
        salt = str(secrets.randbelow(10_000_000_000))
        sign = hashlib.md5(f"{self.app_id}{text}{salt}{self.secret_key}".encode("utf-8")).hexdigest()
        params = urllib.parse.urlencode(
            {
                "q": text,
                "from": source,
                "to": target,
                "appid": self.app_id,
                "salt": salt,
                "sign": sign,
            }
        )
        url = f"{self.endpoint}?{params}"

        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"百度翻译接口 HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"百度翻译接口不可用：{exc.reason}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"百度翻译响应不是 JSON：{body[:500]}") from exc

        if "error_code" in data:
            raise RuntimeError(f"百度翻译错误 {data.get('error_code')}: {data.get('error_msg')}")

        results = data.get("trans_result") or []
        translated = "\n".join(item.get("dst", "") for item in results).strip()
        if not translated:
            raise RuntimeError(f"百度翻译响应缺少 trans_result：{data}")

        return TranslationResult(text=translated, source_lang=source_lang, target_lang=target_lang)

    @staticmethod
    def _normalize_lang(lang: str, is_source: bool) -> str:
        if is_source and lang in {"", "auto", "unknown"}:
            return "auto"
        return {
            "zh": "zh",
            "zh-cn": "zh",
            "zh_cn": "zh",
            "zh-CN": "zh",
            "en": "en",
            "ja": "jp",
            "jp": "jp",
            "ko": "kor",
            "fr": "fra",
            "de": "de",
            "es": "spa",
            "ru": "ru",
        }.get(lang, lang)


class StubTtsEngine(TtsEngine):
    name = "stub-tts"

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    async def synthesize(self, text: str, target_lang: str, voice_id: str) -> TtsResult:
        frequency = 520.0 if target_lang.startswith("en") else 440.0
        return TtsResult(audio=generate_tone_wav(text, self.sample_rate, frequency), sample_rate=self.sample_rate)


class Pyttsx3TtsEngine(TtsEngine):
    name = "pyttsx3-tts"
    _lock = threading.Lock()

    async def synthesize(self, text: str, target_lang: str, voice_id: str) -> TtsResult:
        return await asyncio.to_thread(self._synthesize_sync, text, target_lang, voice_id)

    def _synthesize_sync(self, text: str, target_lang: str, voice_id: str) -> TtsResult:
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError("未安装 pyttsx3。请先运行：pip install -e \".[tts]\"") from exc

        with self._lock:
            try:
                engine = pyttsx3.init()
                self._select_voice(engine, target_lang, voice_id)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    wav_path = Path(tmp_file.name)
                engine.save_to_file(text, str(wav_path))
                engine.runAndWait()
                audio = wav_path.read_bytes()
                sample_rate = self._read_wav_sample_rate(audio)
                return TtsResult(audio=audio, sample_rate=sample_rate, audio_format="wav")
            except Exception:
                return TtsResult(audio=generate_silence_wav(24000, 0.25), sample_rate=24000, audio_format="wav")
            finally:
                try:
                    wav_path.unlink(missing_ok=True)
                except (NameError, OSError):
                    pass

    @staticmethod
    def _select_voice(engine, target_lang: str, voice_id: str) -> None:
        if voice_id and voice_id != "default":
            engine.setProperty("voice", voice_id)
            return

        lang_prefix = target_lang.split("-")[0].lower()
        for voice in engine.getProperty("voices"):
            voice_text = " ".join(
                [
                    str(getattr(voice, "id", "")),
                    str(getattr(voice, "name", "")),
                    str(getattr(voice, "languages", "")),
                ]
            ).lower()
            if lang_prefix in voice_text:
                engine.setProperty("voice", voice.id)
                return

    @staticmethod
    def _read_wav_sample_rate(audio: bytes) -> int:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            return wav_file.getframerate()


class EspeakTtsEngine(TtsEngine):
    name = "espeak-tts"

    def __init__(self, voice: str = "en", speed: int = 165) -> None:
        self.voice = voice
        self.speed = speed

    async def synthesize(self, text: str, target_lang: str, voice_id: str) -> TtsResult:
        return await asyncio.to_thread(self._synthesize_sync, text, target_lang, voice_id)

    def _synthesize_sync(self, text: str, target_lang: str, voice_id: str) -> TtsResult:
        if not text.strip():
            return TtsResult(audio=generate_silence_wav(24000, 0.25), sample_rate=24000, audio_format="wav")

        voice = voice_id if voice_id and voice_id != "default" else self._voice_for_lang(target_lang)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = Path(tmp_file.name)

        command = [
            "espeak-ng",
            "-v",
            voice,
            "-s",
            str(self.speed),
            "-w",
            str(wav_path),
            "--stdin",
        ]
        try:
            subprocess.run(
                command,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            audio = wav_path.read_bytes()
            if not audio:
                return TtsResult(audio=generate_silence_wav(24000, 0.25), sample_rate=24000, audio_format="wav")
            sample_rate = self._safe_read_wav_sample_rate(audio)
            if sample_rate <= 0:
                return TtsResult(audio=generate_silence_wav(24000, 0.25), sample_rate=24000, audio_format="wav")
            return TtsResult(audio=audio, sample_rate=sample_rate, audio_format="wav")
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 espeak-ng。请先运行：apt-get install -y espeak-ng") from exc
        except subprocess.CalledProcessError:
            return TtsResult(audio=generate_silence_wav(24000, 0.25), sample_rate=24000, audio_format="wav")
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _voice_for_lang(self, target_lang: str) -> str:
        lang = target_lang.split("-")[0].lower()
        return {
            "zh": "cmn",
            "en": self.voice,
            "ja": "ja",
            "ko": "ko",
            "fr": "fr",
            "de": "de",
            "es": "es",
            "ru": "ru",
        }.get(lang, self.voice)

    @staticmethod
    def _read_wav_sample_rate(audio: bytes) -> int:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            return wav_file.getframerate()

    @staticmethod
    def _safe_read_wav_sample_rate(audio: bytes) -> int:
        try:
            return EspeakTtsEngine._read_wav_sample_rate(audio)
        except (EOFError, wave.Error):
            return 0


class PassthroughPostProcessor(AudioPostProcessor):
    name = "passthrough-postprocessor"

    async def process(self, audio: bytes, sample_rate: int, audio_format: str) -> TtsResult:
        return TtsResult(audio=audio, sample_rate=sample_rate, audio_format=audio_format)
