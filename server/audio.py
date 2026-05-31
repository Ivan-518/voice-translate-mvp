import base64
import io
import math
import subprocess
import struct
import wave
from array import array


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_base64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def pcm_s16le_duration_seconds(audio: bytes, sample_rate: int, channels: int = 1) -> float:
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    bytes_per_sample = 2
    return len(audio) / float(sample_rate * channels * bytes_per_sample)


def pcm_s16le_to_wav(audio: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
    return buffer.getvalue()


def wav_to_pcm_s16le(audio: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("当前只支持 16-bit PCM WAV")
    if channels == 1:
        return frames, sample_rate

    mono = bytearray()
    frame_width = sample_width * channels
    for offset in range(0, len(frames), frame_width):
        mono.extend(frames[offset : offset + sample_width])
    return bytes(mono), sample_rate


def resample_pcm_s16le(audio: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return audio
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate and target_rate must be positive")
    if not audio:
        return audio

    samples = array("h")
    samples.frombytes(audio)
    if len(samples) == 0:
        return audio

    target_length = max(1, int(len(samples) * target_rate / source_rate))
    converted = array("h")
    converted.extend(0 for _ in range(target_length))

    ratio = source_rate / target_rate
    last_index = len(samples) - 1
    for output_index in range(target_length):
        source_pos = output_index * ratio
        left = int(source_pos)
        right = min(left + 1, last_index)
        fraction = source_pos - left
        value = int(samples[left] * (1.0 - fraction) + samples[right] * fraction)
        converted[output_index] = max(-32768, min(32767, value))

    return converted.tobytes()


def generate_tone_wav(
    text: str,
    sample_rate: int = 24000,
    base_frequency: float = 440.0,
    min_seconds: float = 0.7,
    max_seconds: float = 3.5,
) -> bytes:
    duration = min(max(min_seconds + len(text) * 0.045, min_seconds), max_seconds)
    total_samples = int(sample_rate * duration)
    frames = bytearray()

    for index in range(total_samples):
        t = index / sample_rate
        envelope = min(1.0, index / max(1, sample_rate * 0.04))
        envelope *= min(1.0, (total_samples - index) / max(1, sample_rate * 0.08))
        value = 0.22 * envelope * math.sin(2.0 * math.pi * base_frequency * t)
        frames.extend(struct.pack("<h", int(value * 32767)))

    return pcm_s16le_to_wav(bytes(frames), sample_rate)


def generate_silence_wav(sample_rate: int = 24000, seconds: float = 0.25) -> bytes:
    total_samples = max(1, int(sample_rate * seconds))
    return pcm_s16le_to_wav(b"\x00\x00" * total_samples, sample_rate)


def decode_browser_audio_to_pcm(audio: bytes, input_format: str = "webm", sample_rate: int = 16000) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        input_format,
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 ffmpeg。浏览器录音需要先安装 ffmpeg 并加入 PATH。") from exc
    except subprocess.CalledProcessError as exc:
        error_text = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg 解码浏览器音频失败：{error_text}") from exc

    return completed.stdout
