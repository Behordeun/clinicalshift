import json
import os
from dotenv import load_dotenv
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_DIR = DATA_DIR / "chroma_db"
DB_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_device() -> str:
    """Detect best available device for embedding inference."""
    from clinicalshift import get_device as _get_device
    return _get_device()


def build_collection(name: str, guidelines_path: Path, client, model):
    guidelines = json.loads(guidelines_path.read_text())
    texts = [g["text"] for g in guidelines]
    ids = [g["id"] for g in guidelines]
    print(f"Building collection {name} with {len(texts)} docs...")

    embeddings = model.encode(texts, show_progress_bar=True)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=guidelines,
    )
    return collection


def main():
    device = get_device()
    print(f"Using device: {device}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    build_collection("tau_old", DATA_DIR / "guidelines_tau_old.json", client, model)
    build_collection("tau_new", DATA_DIR / "guidelines_tau_new.json", client, model)
    build_collection("instA", DATA_DIR / "guidelines_instA.json", client, model)
    build_collection("instB", DATA_DIR / "guidelines_instB.json", client, model)

    # Schema erasure: same content as tau_old but with metadata stripped
    schema_erased_path = DATA_DIR / "guidelines_schema_erased.json"
    if schema_erased_path.exists():
        build_collection("schema_erased", schema_erased_path, client, model)
    else:
        # Auto-generate from tau_old by stripping metadata
        tau_old_docs = json.loads(
            (DATA_DIR / "guidelines_tau_old.json").read_text()
        )
        erased = [{"id": g["id"], "text": g["text"]} for g in tau_old_docs]
        schema_erased_path.write_text(json.dumps(erased, indent=2))
        build_collection("schema_erased", schema_erased_path, client, model)

    print("Vector stores built and persisted.")


if __name__ == "__main__":
    main()
