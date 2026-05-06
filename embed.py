import json
from sentence_transformers import SentenceTransformer

FILES = [
    "PT - 3607-32-74-ED-IN-DS-7301-A4-rag_chunks.jsonl",
    "CV - 3607-32-74-ED-IN-DS-7401-A3g-rag_chunks.jsonl"
]

MODEL_PATH = "/mnt/storage-1/home/sadra/AISadra/user-4/e5-embedding/e5-model"

BATCH_SIZE = 32

def run_embedding():
    model = SentenceTransformer(MODEL_PATH)
    
    texts = []
    metadata = []
    
    for file in FILES:

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # اضافه کردن پیشوند passage برای مدل‌های E5 در تولید امبدینگ
                texts.append("passage: " + item["text"])
                metadata.append({"text": item["text"], "source_file": file})
    
    if not texts:
        print("❌ هیچ متنی پیدا نشد")
        return
        
    print(f"تعداد کل تکه‌ها برای امبدینگ: {len(texts)}")
    
    with open("final_embeddings.jsonl", "w", encoding="utf-8") as out:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            for j, emb in enumerate(embeddings):
                out.write(json.dumps({
                    "embedding": emb.tolist(),
                    "metadata": metadata[i + j]
                }, ensure_ascii=False) + "\n")
                
    print("✅ Embedding تمام شد و در final_embeddings.jsonl ذخیره گردید.")

if __name__ == "__main__":
    run_embedding()
