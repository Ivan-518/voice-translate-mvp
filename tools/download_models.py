import argparse
import os
from pathlib import Path

from server.config import get_settings


def configure_hf_endpoint(endpoint: str) -> None:
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)


def download_faster_whisper(model_name: str, device: str, compute_type: str) -> None:
    from faster_whisper import WhisperModel

    print(f"[asr] loading faster-whisper model: {model_name}")
    WhisperModel(model_name, device=device, compute_type=compute_type)
    print("[asr] ready")


def download_nllb(model_name: str) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"[translate] loading NLLB model: {model_name}")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("[translate] ready")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download and cache runtime models.")
    parser.add_argument("--all", action="store_true", help="Download all configured model families.")
    parser.add_argument("--asr", action="store_true", help="Download the configured faster-whisper model.")
    parser.add_argument("--translation", action="store_true", help="Download the configured NLLB model.")
    parser.add_argument("--hf-endpoint", default=settings.hf_endpoint)
    parser.add_argument("--asr-model", default=settings.faster_whisper_model)
    parser.add_argument("--asr-device", default=settings.faster_whisper_device)
    parser.add_argument("--asr-compute-type", default=settings.faster_whisper_compute_type)
    parser.add_argument("--nllb-model", default=settings.nllb_model)
    args = parser.parse_args()

    configure_hf_endpoint(args.hf_endpoint)
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')}")
    print(f"HF_HOME={os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))}")

    download_all = args.all or not (args.asr or args.translation)
    if download_all or args.asr:
        download_faster_whisper(args.asr_model, args.asr_device, args.asr_compute_type)
    if download_all or args.translation:
        download_nllb(args.nllb_model)


if __name__ == "__main__":
    main()

