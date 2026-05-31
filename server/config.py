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
    baidu_asr_api_key: str = Field(default="", validation_alias="BAIDU_ASR_API_KEY")
    baidu_asr_secret_key: str = Field(default="", validation_alias="BAIDU_ASR_SECRET_KEY")
    baidu_asr_cuid: str = Field(default="voice-translate-mvp", validation_alias="BAIDU_ASR_CUID")
    baidu_asr_dev_pid: int = Field(default=1537, validation_alias="BAIDU_ASR_DEV_PID")
    baidu_asr_endpoint: str = Field(default="https://vop.baidu.com/server_api", validation_alias="BAIDU_ASR_ENDPOINT")
    baidu_asr_token_url: str = Field(
        default="https://aip.baidubce.com/oauth/2.0/token",
        validation_alias="BAIDU_ASR_TOKEN_URL",
    )
    baidu_asr_timeout: float = Field(default=15.0, validation_alias="BAIDU_ASR_TIMEOUT")
    baidu_asr_sample_rate: int = Field(default=16000, validation_alias="BAIDU_ASR_SAMPLE_RATE")
    baidu_asr_min_interval: float = Field(default=2.0, validation_alias="BAIDU_ASR_MIN_INTERVAL")
    translation_engine: str = Field(default="stub", validation_alias="TRANSLATION_ENGINE")
    nllb_model: str = Field(default="facebook/nllb-200-distilled-600M", validation_alias="NLLB_MODEL")
    nllb_device: str = Field(default="auto", validation_alias="NLLB_DEVICE")
    nllb_max_new_tokens: int = Field(default=128, validation_alias="NLLB_MAX_NEW_TOKENS")
    qwen_model: str = Field(default="Qwen/Qwen2.5-3B-Instruct", validation_alias="QWEN_MODEL")
    qwen_device: str = Field(default="auto", validation_alias="QWEN_DEVICE")
    qwen_max_new_tokens: int = Field(default=128, validation_alias="QWEN_MAX_NEW_TOKENS")
    llm_translation_base_url: str = Field(default="https://www.packyapi.com/v1", validation_alias="LLM_TRANSLATION_BASE_URL")
    llm_translation_api_key: str = Field(default="", validation_alias="LLM_TRANSLATION_API_KEY")
    llm_translation_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_TRANSLATION_MODEL")
    llm_translation_timeout: float = Field(default=30.0, validation_alias="LLM_TRANSLATION_TIMEOUT")
    llm_translation_temperature: str = Field(default="", validation_alias="LLM_TRANSLATION_TEMPERATURE")
    baidu_translate_app_id: str = Field(default="", validation_alias="BAIDU_TRANSLATE_APP_ID")
    baidu_translate_secret_key: str = Field(default="", validation_alias="BAIDU_TRANSLATE_SECRET_KEY")
    baidu_translate_endpoint: str = Field(
        default="https://fanyi-api.baidu.com/api/trans/vip/translate",
        validation_alias="BAIDU_TRANSLATE_ENDPOINT",
    )
    baidu_translate_timeout: float = Field(default=10.0, validation_alias="BAIDU_TRANSLATE_TIMEOUT")
    tts_engine: str = Field(default="stub", validation_alias="TTS_ENGINE")
    espeak_voice: str = Field(default="en", validation_alias="ESPEAK_VOICE")
    espeak_speed: int = Field(default=165, validation_alias="ESPEAK_SPEED")
    edge_tts_voice: str = Field(default="en-US-JennyNeural", validation_alias="EDGE_TTS_VOICE")
    edge_tts_rate: str = Field(default="+0%", validation_alias="EDGE_TTS_RATE")
    edge_tts_volume: str = Field(default="+0%", validation_alias="EDGE_TTS_VOLUME")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
