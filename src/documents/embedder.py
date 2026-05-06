from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from src.config import DOC_API_KEY, DOC_BASE_URL, DOC_EMBEDDING_MODEL


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Embeddings class compatible with LangChain / FAISS.
    Uses AvalAI / OpenAI compatible embedding API.
    """

    def __init__(self):
        self.embedder = OpenAIEmbeddings(
            model=DOC_EMBEDDING_MODEL,
            api_key=DOC_API_KEY,
            base_url=DOC_BASE_URL
        )

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        return self.embedder.embed_query(text)
