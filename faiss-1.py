import json
import pickle
import numpy as np
import faiss

def build_index():
    embeddings = []
    metadata = []
    
    print("⏳ در حال خواندن فایل امبدینگ‌ها...")
    try:
        with open("final_embeddings.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                embeddings.append(item["embedding"])
                metadata.append(item["metadata"])
    except FileNotFoundError:
        print("❌ فایل final_embeddings.jsonl یافت نشد. لطفاً ابتدا embed.py را اجرا کنید.")
        return
    
    if not embeddings:
        print("❌ هیچ امبدینگی یافت نشد!")
        return
        
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    
    print("embeddings shape:", embeddings.shape)
    print("dimension:", dim)
    
    print("⏳ در حال ساخت ایندکس FAISS...")
    index = faiss.IndexFlatIP(dim) # مناسب برای بردارهای نرمال شده
    index.add(embeddings)
    
    faiss.write_index(index, "vector.index")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print("✅ ایندکس و متادیتا با موفقیت ذخیره شدند.")

if __name__ == "__main__":
    build_index()
