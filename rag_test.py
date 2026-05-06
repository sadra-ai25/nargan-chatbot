#python
from src.documents.rag_engine import DocumentRAGEngine
import asyncio

async def main():
    rag = DocumentRAGEngine()

    print("⏳ Building index...")
    await rag.build_index()

    print("✅ Index built. Running test query...")
    answer = await rag.query("این سندها درباره چه موضوعی هستند؟")
    
    print("\n--- ANSWER ---")
    print(answer)

asyncio.run(main())
