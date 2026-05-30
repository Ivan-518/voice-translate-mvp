from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "Voice Translate MVP"
    default_source_lang: str = Field(default="auto", validation_alias="SOURCE_LANG")
    default_target_lang: str = Field(default="en", validation_alias="TARGET_LANG")
    output_sample_rate: int = Field(default=24000, validation_alias="OUTPUT_SAMPLE_RATE")
    max_audio_seconds: float = Field(default=15.0, validation_alias="MAX_AUDIO_SECONDS")
    hf_endpoint: str = Field(default="https://hf-mirror.com", validation_alias="HF_ENDPOINT")
    asr_engine: str = Field(default="stub", validation_alias="ASR_ENGINE")
    faster_whisper_model: str = Field(default="small", validation_alias="FASTER_WHISPER_MODEL")
    faster_whisper_device: str = Field(default="auto", validation_alias="FASTER_WHISPER_DEVICE")
    faster_whisper_compute_type: str = Field(default="default", validation_alias="FASTER_WHISPER_COMPUTE_TYPE")
    faster_whisper_beam_size: int = Field(default=5, validation_alias="FASTER_WHISPER_BEAM_SIZE")
    faster_whisper_vad_filter: bool = Field(default=True, validation_alias="FASTER_WHISPER_VAD_FILTER")
    faster_whisper_initial_prompt: str = Field(
        default="以下是普通话中文语音，请准确转写为简体中文。",
        validation_alias="FASTER_WHISPER_INITIAL_PROMPT",
    )
    translation_engine: str = Field(default="stub", validation_alias="TRANSLATION_ENGINE")
    nllb_model: str = Field(default="facebook/nllb-200-distilled-600M", validation_alias="NLLB_MODEL")
    nllb_device: str = Field(default="auto", validation_alias="NLLB_DEVICE")
    nllb_max_new_tokens: int = Field(default=128, validation_alias="NLLB_MAX_NEW_TOKENS")
    tts_engine: str = Field(default="stub", validation_alias="TTS_ENGINE")
    espeak_voice: str = Field(default="en", validation_alias="ESPEAK_VOICE")
    espeak_speed: int = Field(default=165, validation_alias="ESPEAK_SPEED")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
