import argparse
import base64
import json
from pathlib import Path
from urllib import request

from client.audio_output import play_wav_bytes


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8010")
    parser.add_argument("--text", default="我马上到")
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--voice-id", default="default")
    parser.add_argument("--output", default="outputs/text_demo.wav")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--output-device", default=None)
    args = parser.parse_args()

    result = post_json(
        f"{args.server}/api/process-text",
        {
            "text": args.text,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "voice_id": args.voice_id,
        },
    )

    audio_format = result.get("audio_format", "wav")
    output_path = output_path_for_format(Path(args.output), audio_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_bytes = base64.b64decode(result["audio_base64"])
    output_path.write_bytes(audio_bytes)

    if args.play:
        if audio_format != "wav":
            raise SystemExit(f"当前命令行播放器只支持 WAV，服务端返回的是 {audio_format}，已保存到 {output_path}")
        play_wav_bytes(audio_bytes, device=args.output_device)

    print("source_text:", result["source_text"])
    print("translated_text:", result["translated_text"])
    print("audio:", output_path)
    print("engine_trace:", " -> ".join(result["engine_trace"]))


def output_path_for_format(path: Path, audio_format: str) -> Path:
    if path.suffix:
        return path.with_suffix(f".{audio_format}")
    return path.with_suffix(f".{audio_format}")


if __name__ == "__main__":
    main()
