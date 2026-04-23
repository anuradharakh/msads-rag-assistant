import json

file_path = "data/processed/canonical_sections.jsonl"

with open(file_path, "r") as f:
    for i, line in enumerate(f):
        record = json.loads(line)

        print("\n--- SAMPLE ---")
        print("TITLE:", record["section_title"])
        print("TYPE:", record["content_type"])
        print("CONTENT:", record["content"][:200])

        if i >= 5:
            break