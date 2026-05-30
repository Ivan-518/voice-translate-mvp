from pydantic import BaseModel, Field


class TextProcessRequest(BaseModel):
    text: str = Field(min_length=1)
    source_lang: str = "auto"
    target_lang: str = "en"
    voice_id: str = "default"


class AudioProcessRequest(BaseModel):
    audio_base64: str = Field(min_length=1)
    sample_rate: int = 16000
    audio_format: str = "pcm_s16le"
    source_lang: str = "auto"
    target_lang: str = "en"
    voice_id: str = "default"


class BrowserAudioProcessRequest(BaseModel):
    audio_base64: str = Field(min_length=1)
    audio_format: str = "webm"
    source_lang: str = "auto"
    target_lang: str = "en"
    voice_id: str = "default"


class SpeechProcessResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    audio_base64: str
    audio_format: str = "wav"
    sample_rate: int
    engine_trace: list[str]
    timings_ms: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    app: str
