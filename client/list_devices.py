def main() -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("请先安装客户端音频依赖：pip install -e \".[client]\"") from exc

    print(sd.query_devices())


if __name__ == "__main__":
    main()

