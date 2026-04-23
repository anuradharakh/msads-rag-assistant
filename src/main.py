from src.pipeline import IngestionPipeline


def main() -> None:
    print("[INFO] Starting ingestion pipeline...")
    pipeline = IngestionPipeline()
    output = pipeline.run()
    print(f"[INFO] Pipeline finished. Output: {output}")


if __name__ == "__main__":
    main()