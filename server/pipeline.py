from dataclasses import dataclass
from time import perf_counter

from server.audio import encode_base64, generate_silence_wav
from server.engines import (
    AsrEngine,
    AudioPostProcessor,
    BaiduAsrEngine,
    BaiduTranslationEngine,
    EdgeTtsEngine,
    EspeakTtsEngine,
    FasterWhisperAsrEngine,
    GoogleTranslationEngine,
    NllbTranslationEngine,
    OpenAICompatibleTranslationEngine,
    PassthroughPostProcessor,
    Pyttsx3TtsEngine,
    QwenTranslationEngine,
    StubAsrEngine,
    StubTranslationEngine,
    StubTtsEngine,
    TranslationEngine,
    TtsEngine,
)
from server.models import SpeechProcessResponse


@dataclass
class SpeechPipeline:
    asr: AsrEngine
    translator: TranslationEngine
    tts: TtsEngine
    postprocessors: list[AudioPostProcessor]
    fallback_sample_rate: int = 24000

    async def process_audio(
        self,
        audio: bytes,
        sample_rate: int,
        source_lang: str,
        target_lang: str,
        voice_id: str,
    ) -> SpeechProcessResponse:
        trace: list[str] = []
        timings: dict[str, int] = {}
        start = perf_counter()
        asr_result = await self.asr.transcribe(audio, sample_rate, source_lang)
        timings["asr"] = int((perf_counter() - start) * 1000)
        trace.append(self.asr.name)
        if not asr_result.text.strip():
            if asr_result.diagnostic:
                trace.append(asr_result.diagnostic)
            silence = generate_silence_wav(self.fallback_sample_rate)
            return SpeechProcessResponse(
                source_text="",
                translated_text="",
                source_lang=asr_result.language,
                target_lang=target_lang,
                audio_base64=encode_base64(silence),
                audio_format="wav",
                sample_rate=self.fallback_sample_rate,
                engine_trace=trace + ["no-speech"],
                timings_ms=timings,
            )
        return await self.process_text(
            text=asr_result.text,
            source_lang=asr_result.language,
            target_lang=target_lang,
            voice_id=voice_id,
            trace=trace,
            timings=timings,
        )

    async def process_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        voice_id: str,
        trace: list[str] | None = None,
        timings: dict[str, int] | None = None,
    ) -> SpeechProcessResponse:
        engine_trace = list(trace or [])
        timing_values = dict(timings or {})
        start = perf_counter()
        translation = await self.translator.translate(text, source_lang, target_lang)
        timing_values["translation"] = int((perf_counter() - start) * 1000)
        engine_trace.append(self.translator.name)

        start = perf_counter()
        tts_result = await self.tts.synthesize(translation.text, target_lang, voice_id)
        timing_values["tts"] = int((perf_counter() - start) * 1000)
        engine_trace.append(self.tts.name)

        audio = tts_result.audio
        sample_rate = tts_result.sample_rate
        audio_format = tts_result.audio_format
        for processor in self.postprocessors:
            start = perf_counter()
            processed = await processor.process(audio, sample_rate, audio_format)
            timing_values[processor.name] = int((perf_counter() - start) * 1000)
            audio = processed.audio
            sample_rate = processed.sample_rate
            audio_format = processed.audio_format
            engine_trace.append(processor.name)

        return SpeechProcessResponse(
            source_text=text,
            translated_text=translation.text,
            source_lang=translation.source_lang,
            target_lang=translation.target_lang,
            audio_base64=encode_base64(audio),
            audio_format=audio_format,
            sample_rate=sample_rate,
            engine_trace=engine_trace,
            timings_ms=timing_values,
        )


def create_default_pipeline(
    output_sample_rate: int,
    asr_engine: str = "stub",
    faster_whisper_model: str = "small",
    faster_whisper_device: str = "auto",
    faster_whisper_compute_type: str = "default",
    faster_whisper_beam_size: int = 5,
    faster_whisper_vad_filter: bool = True,
    faster_whisper_initial_prompt: str = "",
    baidu_asr_api_key: str = "",
    baidu_asr_secret_key: str = "",
    baidu_asr_cuid: str = "voice-translate-mvp",
    baidu_asr_dev_pid: int = 1537,
    baidu_asr_endpoint: str = "https://vop.baidu.com/server_api",
    baidu_asr_token_url: str = "https://aip.baidubce.com/oauth/2.0/token",
    baidu_asr_timeout: float = 15.0,
    baidu_asr_sample_rate: int = 16000,
    baidu_asr_min_interval: float = 2.0,
    translation_engine: str = "stub",
    nllb_model: str = "facebook/nllb-200-distilled-600M",
    nllb_device: str = "auto",
    nllb_max_new_tokens: int = 128,
    qwen_model: str = "Qwen/Qwen2.5-3B-Instruct",
    qwen_device: str = "auto",
    qwen_max_new_tokens: int = 128,
    llm_translation_base_url: str = "https://www.packyapi.com/v1",
    llm_translation_api_key: str = "",
    llm_translation_model: str = "gpt-4o-mini",
    llm_translation_timeout: float = 30.0,
    llm_translation_temperature: str = "",
    baidu_translate_app_id: str = "",
    baidu_translate_secret_key: str = "",
    baidu_translate_endpoint: str = "https://fanyi-api.baidu.com/api/trans/vip/translate",
    baidu_translate_timeout: float = 10.0,
    tts_engine: str = "stub",
    espeak_voice: str = "en",
    espeak_speed: int = 165,
    edge_tts_voice: str = "en-US-JennyNeural",
    edge_tts_rate: str = "+0%",
    edge_tts_volume: str = "+0%",
) -> SpeechPipeline:
    if asr_engine == "stub":
        asr = StubAsrEngine()
    elif asr_engine == "faster_whisper":
        asr = FasterWhisperAsrEngine(
            model_name=faster_whisper_model,
            device=faster_whisper_device,
            compute_type=faster_whisper_compute_type,
            beam_size=faster_whisper_beam_size,
            vad_filter=faster_whisper_vad_filter,
            initial_prompt=faster_whisper_initial_prompt,
        )
    elif asr_engine == "baidu":
        asr = BaiduAsrEngine(
            api_key=baidu_asr_api_key,
            secret_key=baidu_asr_secret_key,
            cuid=baidu_asr_cuid,
            dev_pid=baidu_asr_dev_pid,
            endpoint=baidu_asr_endpoint,
            token_url=baidu_asr_token_url,
            timeout=baidu_asr_timeout,
            sample_rate=baidu_asr_sample_rate,
            min_interval=baidu_asr_min_interval,
        )
    else:
        raise ValueError(f"unsupported ASR_ENGINE: {asr_engine}")

    if translation_engine == "stub":
        translator = StubTranslationEngine()
    elif translation_engine == "google":
        translator = GoogleTranslationEngine()
    elif translation_engine == "nllb":
        translator = NllbTranslationEngine(
            model_name=nllb_model,
            device=nllb_device,
            max_new_tokens=nllb_max_new_tokens,
        )
    elif translation_engine == "qwen":
        translator = QwenTranslationEngine(
            model_name=qwen_model,
            device=qwen_device,
            max_new_tokens=qwen_max_new_tokens,
        )
    elif translation_engine in {"llm", "openai"}:
        translator = OpenAICompatibleTranslationEngine(
            base_url=llm_translation_base_url,
            api_key=llm_translation_api_key,
            model=llm_translation_model,
            timeout=llm_translation_timeout,
            temperature=llm_translation_temperature,
        )
    elif translation_engine == "baidu":
        translator = BaiduTranslationEngine(
            app_id=baidu_translate_app_id,
            secret_key=baidu_translate_secret_key,
            endpoint=baidu_translate_endpoint,
            timeout=baidu_translate_timeout,
        )
    else:
        raise ValueError(f"unsupported TRANSLATION_ENGINE: {translation_engine}")

    if tts_engine == "stub":
        tts = StubTtsEngine(sample_rate=output_sample_rate)
    elif tts_engine == "pyttsx3":
        tts = Pyttsx3TtsEngine()
    elif tts_engine == "espeak":
        tts = EspeakTtsEngine(voice=espeak_voice, speed=espeak_speed)
    elif tts_engine == "edge":
        tts = EdgeTtsEngine(voice=edge_tts_voice, rate=edge_tts_rate, volume=edge_tts_volume)
    else:
        raise ValueError(f"unsupported TTS_ENGINE: {tts_engine}")

    return SpeechPipeline(
        asr=asr,
        translator=translator,
        tts=tts,
        postprocessors=[PassthroughPostProcessor()],
        fallback_sample_rate=output_sample_rate,
    )
