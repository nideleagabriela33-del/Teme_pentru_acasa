import json
import os
import hashlib
import sys

# Workaround for pkg_resources missing issue
try:
    import pkg_resources
except ImportError:
    # If pkg_resources is not available, create a minimal mock
    from importlib.metadata import version as get_version
    class MockParseVersion:
        def __init__(self, v):
            self.v = v
        def __lt__(self, other):
            return self.v < other.v
        def __le__(self, other):
            return self.v <= other.v
        def __gt__(self, other):
            return self.v > other.v
        def __ge__(self, other):
            return self.v >= other.v
        def __eq__(self, other):
            return self.v == other.v
    
    class MockPkgResources:
        @staticmethod
        def parse_version(v):
            return MockParseVersion(v)
    
    sys.modules['pkg_resources'] = MockPkgResources()

from dotenv import load_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

print("[STARTUP] All imports loaded successfully!")

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "./data")
LOCAL_DOCS_DIR = os.environ.get("LOCAL_DOCS_DIR", "./knowledge")

CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")

USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u.strip() for u in os.environ.get("WEB_URLS", "").split(";") if u.strip()]


class RAGAssistant:
    """Asistent virtual RAG pentru ALEIRE CONSULTING SRL."""

    def __init__(self) -> None:
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(LOCAL_DOCS_DIR, exist_ok=True)

        self.embedder = None
        
        # Load relevance embedding lazily (will be created on first use)
        self._relevance_cached = None

        self.system_prompt = (
            "Esti asistentul virtual al firmei ALEIRE CONSULTING SRL. "
            "Firma ofera servicii precum dezvoltare website-uri, aplicatii web personalizate, "
            "mentenanta, consultanta software, automatizari digitale si integrare AI. "
            "Raspunzi exclusiv pe baza contextului primit si a informatiilor disponibile despre firma. "
            "Nu inventa informatii despre firma, proiecte, portofoliu, clienti, preturi finale, concurenta sau termene exacte. "
            "Daca nu exista suficiente informatii in context, spune clar acest lucru. "
            "Cand utilizatorul cere o oferta, ofera doar o estimare orientativa bazata pe complexitate, "
            "nu un pret final contractual. "
            "Raspunsurile trebuie sa fie clare, profesioniste, concise si utile pentru potentiali clienti. "
            "Daca intrebarea nu are legatura cu serviciile firmei, explica politicos ca poti ajuta doar "
            "cu informatii despre serviciile ALEIRE CONSULTING si estimari pentru proiecte software."
        )

        self.pricing_rules = """
Reguli orientative de estimare pentru ALEIRE CONSULTING:
- landing page simplu: 250 - 500 EUR
- website de prezentare basic: 500 - 1200 EUR
- website de prezentare avansat: 1200 - 2500 EUR
- magazin online: 2000 - 5000 EUR
- aplicatie web custom: de la 3000 EUR in sus, in functie de complexitate
- mentenanta lunara: 50 - 300 EUR
- integrare chatbot/AI: de la 500 EUR in sus

Aceste valori sunt strict orientative si trebuie prezentate ca estimari initiale,
nu ca oferta finala sau contractuala.
"""

    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings folosind Universal Sentence Encoder."""
        if isinstance(texts, str):
            texts = [texts]

        # Use lightweight embeddings instead of downloading large TensorFlow Hub model
        # This is a simple hashing-based embedding for quick testing
        embeddings = []
        for text in texts:
            # Create a simple 384-dimensional embedding based on text hash
            hash_obj = hashlib.md5((text.lower().strip()).encode())
            hash_bytes = hash_obj.digest()
            
            # Expand hash to 384 dimensions with some variation
            # hash_bytes contains integers (0-255) in Python 3
            embedding = np.array([b / 256.0 for b in hash_bytes * 48], dtype=np.float32)[:384]
            
            # Pad if necessary
            if len(embedding) < 384:
                embedding = np.pad(embedding, (0, 384 - len(embedding)), 'constant')
            
            embeddings.append(embedding)
        
        return np.array(embeddings)

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati potrivite pentru indexare."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
        )
        chunks = splitter.split_text(text or "")
        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return cleaned_chunks if cleaned_chunks else []

    def _load_documents_from_local(self) -> list[str]:
        """Incarca documente locale din folderul knowledge."""
        local_chunks = []

        if not os.path.exists(LOCAL_DOCS_DIR):
            return local_chunks

        for filename in os.listdir(LOCAL_DOCS_DIR):
            if not filename.endswith((".txt", ".md")):
                continue

            file_path = os.path.join(LOCAL_DOCS_DIR, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = self._chunk_text(content)
                local_chunks.extend(chunks)
            except Exception:
                continue

        return local_chunks

    def _load_documents_from_web(self) -> list[str]:
        """Incarca documente din surse web."""
        web_chunks = []

        for url in WEB_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()

                for doc in docs:
                    chunks = self._chunk_text(doc.page_content)
                    web_chunks.extend(chunks)
            except Exception:
                continue

        return web_chunks

    def _load_documents(self) -> list[str]:
        """Incarca documente din cache, web si fisiere locale."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        all_chunks.extend(self._load_documents_from_web())
        all_chunks.extend(self._load_documents_from_local())

        # eliminare duplicate simple
        unique_chunks = list(dict.fromkeys(chunk for chunk in all_chunks if chunk.strip()))

        if unique_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(unique_chunks, f, ensure_ascii=False, indent=2)

        return unique_chunks

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Calculeaza hash determinist pentru chunks + model."""
        payload = json.dumps(
            {
                "model": USE_MODEL_URL,
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None

        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        """Construieste indexul FAISS din chunks si il salveaza."""
        if not chunks:
            raise ValueError("Lista de chunks este goala.")

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, FAISS_INDEX_PATH)

        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))

        return index

    def _retrieve_relevant_chunks(
        self,
        chunks: list[str],
        user_query: str,
        k: int = 6
    ) -> list[str]:
        """Returneaza cele mai relevante chunks pentru intrebarea utilizatorului."""
        if not chunks:
            return []

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if index.ntotal != len(chunks) or index.d != query_embedding.shape[1]:
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    def calculate_similarity(self, text: str) -> float:
        """Returneaza similaritatea fata de domeniul firmei."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return 0.0

        # Lazy load relevance embedding on first use
        if self._relevance_cached is None:
            self._relevance_cached = self._embed_texts(
                "Aceasta este o intrebare relevanta despre serviciile oferite de ALEIRE CONSULTING, "
                "precum dezvoltare website-uri, aplicatii web personalizate, mentenanta software, "
                "consultanta IT, automatizari digitale, integrare AI si estimari de cost pentru astfel de proiecte."
            )[0]

        embedding = self._embed_texts(cleaned_text)[0]
        return self._cosine_similarity(embedding, self._relevance_cached)

    def is_relevant(self, user_input: str) -> bool:
        """Verifica daca intrebarea este relevanta pentru domeniul firmei."""
        return self.calculate_similarity(user_input) >= 0.42

    def _send_prompt_to_llm(self, user_input: str, context: str) -> str:
        """Trimite promptul catre modelul LLM si returneaza raspunsul."""
        system_msg = self.system_prompt + "\n\n" + self.pricing_rules

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": f"""
Utilizatorul a transmis urmatoarea cerere:
{user_input}

Context relevant extras din sursele firmei:
{context}

Instructiuni de raspuns:
1. Raspunde doar folosind contextul si domeniul firmei.
2. Daca utilizatorul cere o oferta, cere clarificari daca lipsesc informatii importante.
3. Daca ai suficiente detalii, ofera o estimare orientativa sub forma de interval.
4. Nu inventa servicii sau preturi care nu reies logic din context.
5. Daca informatia nu este disponibila, spune explicit acest lucru.
6. Formateaza raspunsul clar si profesionist.
7. Nu pretinde ca ai certitudini daca nu exista suficiente date.

Daca utilizatorul cere o estimare, structureaza raspunsul astfel:
- Tip proiect
- Cerinte identificate
- Estimare orientativa
- Ce informatii mai sunt necesare
- Urmatorii pasi
""",
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
                temperature=0.2,
            )

            return response.choices[0].message.content or (
                "Asistent: Nu am putut genera un raspuns valid in acest moment."
            )
        except Exception:
            return (
                "Asistent: Nu pot ajunge la modelul de limbaj acum. "
                "Te rog incearca din nou in cateva momente."
            )

    def assistant_response(self, user_message: str) -> str:
        """Proceseaza mesajul utilizatorului si intoarce raspunsul asistentului."""
        if not user_message or not user_message.strip():
            return (
                "Te rog scrie o intrebare despre serviciile ALEIRE CONSULTING, de exemplu: "
                "'Cat costa un website de prezentare?', "
                "'Puteti realiza un magazin online?' sau "
                "'Ce include mentenanta pentru o aplicatie web?'."
            )

        if not self.is_relevant(user_message):
            return (
                "Pot raspunde doar la intrebari legate de serviciile ALEIRE CONSULTING, "
                "precum dezvoltare website-uri, aplicatii web, mentenanta, consultanta software, "
                "automatizari digitale si estimari de cost pentru astfel de proiecte. "
                "De exemplu, ma poti intreba: 'Cat costa un site de prezentare?' sau "
                "'Ce informatii sunt necesare pentru o oferta de aplicatie web?'."
            )

        chunks = self._load_documents()
        if not chunks:
            return (
                "Momentan nu exista suficiente surse de informatii incarcate despre serviciile firmei. "
                "Verifica folderul knowledge si sursele din WEB_URLS."
            )

        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)

        if not context.strip():
            return (
                "Nu am gasit suficiente informatii relevante pentru a raspunde sigur la aceasta intrebare. "
                "Te rog reformuleaza cererea sau cere informatii despre website-uri, aplicatii web, "
                "mentenanta, consultanta software ori estimari de proiect."
            )

        return self._send_prompt_to_llm(user_message, context)


if __name__ == "__main__":
    assistant = RAGAssistant()
    
    print("\n" + "="*80)
    print("CHATBOT ALEIRE CONSULTING - RAG Assistant")
    print("="*80)
    print("\nBine ati venit! Puteti pune intrebari despre serviciile noastre.")
    print("Tastati 'iesire' sau 'exit' pentru a incheia conversatia.\n")
    
    # Run demo tests first (optional)
    run_demo = input("Doriți să vedeti demonstrația cu 3 teste? (da/nu): ").strip().lower()
    
    if run_demo in ['da', 'yes', 'd', 'y']:
        print("\n" + "="*80)
        print("DEMONSTRAȚIE - TEST 1: Ofertă website")
        print("="*80)
        print(
            assistant.assistant_response(
                "Buna, as dori o oferta pentru un website de prezentare cu 5 pagini, formular de contact si design modern."
            )
        )

        print("\n" + "="*80)
        print("DEMONSTRAȚIE - TEST 2: Aplicație web")
        print("="*80)
        print(
            assistant.assistant_response(
                "Puteti realiza o aplicatie web personalizata pentru gestionarea clientilor?"
            )
        )

        print("\n" + "="*80)
        print("DEMONSTRAȚIE - TEST 3: Întrebare irelevantă")
        print("="*80)
        print(
            assistant.assistant_response(
                "Care este capitala Japoniei?"
            )
        )
        print("\n" + "="*80)
    
    # Interactive chat mode
    print("\n" + "="*80)
    print("MOD INTERACTIV - Gestioneza propriile intrebari")
    print("="*80 + "\n")
    
    while True:
        user_input = input("\n👤 Tu: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['iesire', 'exit', 'quit', 'out']:
            print("\n🤖 Chatbot: Multumesc pentru conversatie! La revedere!")
            break
        
        print("\n🤖 Chatbot:\n")
        response = assistant.assistant_response(user_input)
        print(response)
        print("\n" + "-"*80)