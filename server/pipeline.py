from dataclasses import dataclass
from time import perf_counter

from server.audio import encode_base64, generate_silence_wav
from server.engines import (
    AsrEngine,
    AudioPostProcessor,
    FasterWhisperAsrEngine,
    GoogleTranslationEngine,
    PassthroughPostProcessor,
    Pyttsx3TtsEngine,
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
    translation_engine: str = "stub",
    tts_engine: str = "stub",
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
    else:
        raise ValueError(f"unsupported ASR_ENGINE: {asr_engine}")

    if translation_engine == "stub":
        translator = StubTranslationEngine()
    elif translation_engine == "google":
        translator = GoogleTranslationEngine()
    else:
        raise ValueError(f"unsupported TRANSLATION_ENGINE: {translation_engine}")

    if tts_engine == "stub":
        tts = StubTtsEngine(sample_rate=output_sample_rate)
    elif tts_engine == "pyttsx3":
        tts = Pyttsx3TtsEngine()
    else:
        raise ValueError(f"unsupported TTS_ENGINE: {tts_engine}")

    return SpeechPipeline(
        asr=asr,
        translator=translator,
        tts=tts,
        postprocessors=[PassthroughPostProcessor()],
        fallback_sample_rate=output_sample_rate,
    )
