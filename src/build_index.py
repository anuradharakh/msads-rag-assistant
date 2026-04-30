from src.indexer import VectorIndexer


def main() -> None:
    print("[INFO] Starting Phase 2: Chunking + Vector Indexing")
    indexer = VectorIndexer()
    indexer.build_index()
    print("[INFO] Phase 2 indexing complete")


if __name__ == "__main__":
    main()