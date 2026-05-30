import asyncio
import io
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

from server.audio import generate_silence_wav, generate_tone_wav, pcm_s16le_duration_seconds


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str


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
            text,
        ]
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio = wav_path.read_bytes()
            sample_rate = self._read_wav_sample_rate(audio)
            return TtsResult(audio=audio, sample_rate=sample_rate, audio_format="wav")
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 espeak-ng。请先运行：apt-get install -y espeak-ng") from exc
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


class PassthroughPostProcessor(AudioPostProcessor):
    name = "passthrough-postprocessor"

    async def process(self, audio: bytes, sample_rate: int, audio_format: str) -> TtsResult:
        return TtsResult(audio=audio, sample_rate=sample_rate, audio_format=audio_format)
