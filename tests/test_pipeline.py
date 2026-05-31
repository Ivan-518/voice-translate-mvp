import base64
import asyncio

from server.engines import AsrResult, BaiduAsrEngine
from server.pipeline import create_default_pipeline


def test_process_text_returns_audio() -> None:
    pipeline = create_default_pipeline(output_sample_rate=24000)

    result = asyncio.run(
        pipeline.process_text(
            text="我马上到",
            source_lang="zh",
            target_lang="en",
            voice_id="default",
        )
    )

    assert result.translated_text == "I'll be there soon."
    assert result.audio_format == "wav"
    assert result.sample_rate == 24000
    assert base64.b64decode(result.audio_base64).startswith(b"RIFF")
    assert result.engine_trace == ["stub-translation", "stub-tts", "passthrough-postprocessor"]


def test_baidu_asr_local_rate_limit_skips_fast_requests() -> None:
    engine = BaiduAsrEngine(api_key="api-key", secret_key="secret-key", min_interval=10.0)

    assert engine._should_skip_for_rate_limit() is False
    assert engine._should_skip_for_rate_limit() is True


def test_asr_diagnostic_is_exposed_in_engine_trace() -> None:
    class EmptyAsr:
        name = "baidu-asr"

        async def transcribe(self, audio: bytes, sample_rate: int, source_lang: str) -> AsrResult:
            return AsrResult(text="", language="zh", diagnostic="baidu-asr-rate-limited-3305")

    pipeline = create_default_pipeline(output_sample_rate=24000)
    pipeline.asr = EmptyAsr()

    result = asyncio.run(
        pipeline.process_audio(
            audio=b"\x00\x00",
            sample_rate=16000,
            source_lang="zh",
            target_lang="en",
            voice_id="default",
        )
    )

    assert result.engine_trace == ["baidu-asr", "baidu-asr-rate-limited-3305", "no-speech"]
