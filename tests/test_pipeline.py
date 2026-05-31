import base64
import asyncio

from server.engines import BaiduAsrEngine
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
