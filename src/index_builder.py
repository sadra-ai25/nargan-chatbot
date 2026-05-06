"""
FAISS index builder script
Usage: python -m src.index_builder
"""
import json
import pickle
import numpy as np
import faiss

from src.config import EMBEDDINGS_PATH, INDEX_PATH, METADATA_PATH, DATA_DIR


def build_index():
    """Build FAISS index from embeddings"""
    print("⏳ Reading embeddings file...")

    if not EMBEDDINGS_PATH.exists():
        print(f"❌ Embeddings file not found: {EMBEDDINGS_PATH}")
        print("   Please run 'python -m src.embed' first.")
        return

    embeddings = []
    metadata = []

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                embeddings.append(item["embedding"])
                metadata.append(item["metadata"])
            except json.JSONDecodeError:
                print(f"  ⚠️  Skipping invalid JSON at line {line_num}")

    if not embeddings:
        print("❌ No embeddings found!")
        return

    # Convert to numpy array
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"📐 Dimension: {dim}")
    print(f"📁 Total documents: {len(metadata)}")

    # Build FAISS index (Inner Product for normalized vectors)
    print("⏳ Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index and metadata
    print("💾 Saving index and metadata...")
    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Index saved to: {INDEX_PATH}")
    print(f"✅ Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    build_index()