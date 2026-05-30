import io
import wave


def play_wav_bytes(wav_bytes: bytes, device: str | int | None = None) -> None:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as exc:
        raise SystemExit("请先安装客户端音频依赖：pip install -e \".[client]\"") from exc

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("当前客户端只支持 16-bit PCM WAV 播放")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()

