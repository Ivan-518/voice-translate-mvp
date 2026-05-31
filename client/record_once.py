import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets

from client.audio_output import play_wav_bytes


def record_pcm(seconds: float, sample_rate: int) -> bytes:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as exc:
        raise SystemExit("请先安装客户端依赖：pip install -e \".[client]\"") from exc

    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    return np.asarray(audio).reshape(-1).tobytes()


async def run(args: argparse.Namespace) -> None:
    audio = record_pcm(args.seconds, args.sample_rate)
    async with websockets.connect(args.ws_url) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "config",
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                    "voice_id": args.voice_id,
                }
            )
        )
        await websocket.recv()

        await websocket.send(
            json.dumps(
                {
                    "type": "audio_chunk",
                    "chunk_id": 1,
                    "sample_rate": args.sample_rate,
                    "audio_format": "pcm_s16le",
                    "audio_base64": base64.b64encode(audio).decode("ascii"),
                }
            )
        )
        response = json.loads(await websocket.recv())

    payload = response["payload"]
    audio_format = payload.get("audio_format", "wav")
    output_path = output_path_for_format(Path(args.output), audio_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_bytes = base64.b64decode(payload["audio_base64"])
    output_path.write_bytes(audio_bytes)

    if args.play:
        if audio_format != "wav":
            raise SystemExit(f"当前命令行播放器只支持 WAV，服务端返回的是 {audio_format}，已保存到 {output_path}")
        play_wav_bytes(audio_bytes, device=args.output_device)

    print("source_text:", payload["source_text"])
    print("translated_text:", payload["translated_text"])
    print("audio:", output_path)
    print("engine_trace:", " -> ".join(payload["engine_trace"]))


def output_path_for_format(path: Path, audio_format: str) -> Path:
    if path.suffix:
        return path.with_suffix(f".{audio_format}")
    return path.with_suffix(f".{audio_format}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8010/ws/speech")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--voice-id", default="default")
    parser.add_argument("--output", default="outputs/record_once.wav")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--output-device", default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
