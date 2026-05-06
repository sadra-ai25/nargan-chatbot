"""
Embedding generation script
Usage: python -m src.embed
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.config import (
    MODEL_PATH,
    SOURCE_FILES,
    EMBEDDINGS_PATH,
    EMBEDDING_BATCH_SIZE,
    DATA_DIR
)


def run_embedding():
    """Generate embeddings for all source documents"""
    print("⏳ Loading embedding model...")
    model = SentenceTransformer(MODEL_PATH)

    texts = []
    metadata = []

    # Load documents from source files
    for file_path in SOURCE_FILES:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        print(f"📄 Reading: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    # Add E5 prefix for passage embedding
                    texts.append("passage: " + item["text"])
                    metadata.append({
                        "text": item["text"],
                        "source_file": str(file_path.name),
                        "line": line_num
                    })
                except json.JSONDecodeError:
                    print(f"  ⚠️  Skipping invalid JSON at line {line_num}")

    if not texts:
        print("❌ No texts found")
        return

    print(f"📊 Total chunks for embedding: {len(texts)}")

    # Generate embeddings in batches
    print("⏳ Generating embeddings...")
    output_path = DATA_DIR / "final_embeddings.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as out:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            for j, emb in enumerate(embeddings):
                out.write(json.dumps({
                    "embedding": emb.tolist(),
                    "metadata": metadata[i + j]
                }, ensure_ascii=False) + "\n")

            # Progress indicator
            processed = min(i + EMBEDDING_BATCH_SIZE, len(texts))
            print(f"  Progress: {processed}/{len(texts)} chunks")

    print(f"✅ Embedding complete! Saved to: {output_path}")


if __name__ == "__main__":
    run_embedding()