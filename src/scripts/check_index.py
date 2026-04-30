from src.vector_store import ChromaVectorStore


def main() -> None:
    store = ChromaVectorStore()
    count = store.count()

    print(f"Chroma collection count: {count}")

    if count == 0:
        print("Index is empty. Run:")
        print("python -m src.build_index")
    else:
        print("Index is ready.")


if __name__ == "__main__":
    main()