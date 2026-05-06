import re
import unicodedata
from pathlib import Path
from typing import List, Callable

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from src.config import (
    DOC_API_KEY,
    DOC_LLM_MODEL,
    DOC_BASE_URL,
    DOCUMENTS_PDF_DIR,
    DOCUMENTS_INDEX_DIR,
)
from .embedder import LocalSentenceTransformerEmbeddings


FA_TO_EN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def clean_fa_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(FA_TO_EN_DIGITS)

    text = re.sub(r"[\uE000-\uF8FF]", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def soft_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[•▪◦]+", "-", text)
    return text.strip()


GARBLED_CHARS = set("ؼًٍٍَُ")


def is_garbled(text: str) -> bool:
    if not text:
        return True

    bad = sum(1 for c in text if c in GARBLED_CHARS)
    return (bad / max(len(text), 1)) > 0.02


def keyword_boost(docs: List, query: str) -> List:

    q = clean_fa_text(query)
    keywords = [w for w in q.split() if len(w) >= 3]

    boosted = []

    for d in docs:
        hits = sum(1 for k in keywords if k in d.page_content)

        if hits:
            d.metadata["keyword_hits"] = hits
            boosted.append(d)

    return boosted


STRICT_PROMPT = """
شما یک دستیار هستید که فقط و فقط بر اساس متن ارائه شده (Context) پاسخ می‌دهید.

قوانین سخت‌گیرانه:
۱. فقط از متن‌های داخل Context استفاده کن.
۲. اگر هیچ متن مرتبطی وجود ندارد، دقیقاً بگو:
"منبع مجاز برای این پرسش در دسترس نیست. لطفاً استاندارد یا بخش دیگری ارائه دهید."
۳. اگر چند بخش مرتبط وجود دارد:
   - ابتدا همه‌ی بخش‌ها را لیست کن
   - برای هر بخش نام سند و شماره صفحه را ذکر کن
۴. بعد از لیست منابع یک جمع‌بندی بنویس.

Context:
{context}

Question:
{input}

Answer:
"""


class DocumentRAGEngine:

    def __init__(self, config: dict = None, progress_hook: Callable = None):

        self.config = config or {}
        self.progress_hook = progress_hook or (lambda *_: None)

        self.embedding_model = LocalSentenceTransformerEmbeddings()

        self.llm = ChatOpenAI(
            model=DOC_LLM_MODEL,
            api_key=DOC_API_KEY,
            base_url=DOC_BASE_URL,
            temperature=0
            )

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.all_chunks = []

    # ============================================================
    # LOAD ENGINE (FIXED)
    # ============================================================

    async def load(self):

        if DOCUMENTS_INDEX_DIR.exists():

            try:

                self.progress_hook("Loading existing FAISS index...")

                self.vectorstore = FAISS.load_local(
                    folder_path=str(DOCUMENTS_INDEX_DIR),
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                )

                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 40}
                )

                self._build_chain()

                self.progress_hook("✅ Document RAG loaded")

                return

            except Exception as e:

                self.progress_hook(f"Index load failed: {e}")

        self.progress_hook("No index found, building new index...")

        await self.build_index()

    # ============================================================
    # BUILD INDEX
    # ============================================================

    async def build_index(self):

        self.progress_hook("Loading PDFs...")
        documents = self._load_pdfs()

        self.progress_hook("Chunking documents...")
        chunks = self._chunk_documents(documents)

        self.all_chunks = chunks

        self.progress_hook("Creating FAISS index...")

        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embedding_model
        )

        DOCUMENTS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(str(DOCUMENTS_INDEX_DIR))

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 40}
        )

        self._build_chain()

        self.progress_hook("✅ RAG system ready.")

    # ============================================================
    # QUERY
    # ============================================================

    async def query(self, question: str):

        vector_docs = self.retriever.invoke(question)

        keyword_docs = keyword_boost(self.all_chunks, question)

        doc_map = {}

        for d in vector_docs + keyword_docs:
            key = (d.metadata["pdf_name"], d.metadata["page_number"])
            doc_map[key] = d

        found_docs = list(doc_map.values())

        found_docs = sorted(
            found_docs,
            key=lambda d: (
                d.metadata["pdf_name"],
                d.metadata["page_number"]
            )
        )

        found_docs = found_docs[:12]

        if not found_docs:
            return "منبع مجاز برای این پرسش در دسترس نیست."

        result = self.rag_chain.invoke({
            "input": question,
            "context": found_docs
        })

        return result["answer"].strip()

    # ============================================================
    # LOAD PDFs
    # ============================================================

    def _load_pdfs(self):

        documents = []

        for pdf in DOCUMENTS_PDF_DIR.glob("*.pdf"):

            loader = PyMuPDFLoader(str(pdf))
            pages = loader.load()

            for d in pages:

                raw = clean_fa_text(d.page_content)

                if is_garbled(raw):
                    continue

                d.page_content = raw

                d.metadata["pdf_name"] = Path(
                    d.metadata.get("source", "")
                ).name

                d.metadata["page_number"] = (
                    d.metadata.get("page", 0) + 1
                )

                documents.append(d)

        if not documents:
            raise RuntimeError("No valid text found in PDFs")

        return documents

    # ============================================================
    # CHUNK
    # ============================================================

    def _chunk_documents(self, documents):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        docs = splitter.split_documents(documents)

        for d in docs:
            d.page_content = soft_clean(d.page_content)

        return docs

    # ============================================================
    # BUILD CHAIN
    # ============================================================

    def _build_chain(self):

        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=STRICT_PROMPT
        )

        document_prompt = PromptTemplate(
            input_variables=["page_content", "pdf_name", "page_number"],
            template="[سند: {pdf_name} | صفحه: {page_number}]\n{page_content}"
        )

        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=document_prompt
        )

        self.rag_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )


doc_rag_engine = DocumentRAGEngine()
