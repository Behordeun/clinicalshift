import hashlib
import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_DIR = DATA_DIR / "chroma_db"
DB_DIR.mkdir(exist_ok=True, parents=True)

# Stores content hashes to detect when re-embedding is unnecessary
HASH_FILE = DB_DIR / ".collection_hashes.json"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_device() -> str:
    """Detect best available device for embedding inference."""
    from clinicalshift import get_device as _get_device

    return _get_device()


def _compute_hash(guidelines_path: Path, model_name: str) -> str:
    """Hash the guideline content + model name to detect changes."""
    content = guidelines_path.read_bytes()
    h = hashlib.sha256(content)
    h.update(model_name.encode())
    return h.hexdigest()[:16]


def _load_hashes() -> dict:
    if HASH_FILE.exists():
        return json.loads(HASH_FILE.read_text())
    return {}


def _save_hashes(hashes: dict):
    HASH_FILE.write_text(json.dumps(hashes, indent=2))


def build_collection(name: str, guidelines_path: Path, client, model, force: bool = False):
    """Build or rebuild a ChromaDB collection from a guideline JSON file.

    Skips re-embedding if the source content and model haven't changed
    (detected via content hash), unless force=True.
    """
    current_hash = _compute_hash(guidelines_path, EMBED_MODEL_NAME)
    stored_hashes = _load_hashes()

    if not force and stored_hashes.get(name) == current_hash:
        # Check the collection actually exists in ChromaDB
        existing_names = [c.name for c in client.list_collections()]
        if name in existing_names:
            coll = client.get_collection(name)
            print(f"Skipping {name}: unchanged ({coll.count()} docs, hash={current_hash})")
            return coll

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

    # Update hash
    stored_hashes[name] = current_hash
    _save_hashes(stored_hashes)

    return collection


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB vector stores.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if content hasn't changed",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Embedding model: {EMBED_MODEL_NAME}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    build_collection(
        "tau_old", DATA_DIR / "guidelines_tau_old.json", client, model, force=args.force
    )
    build_collection(
        "tau_new", DATA_DIR / "guidelines_tau_new.json", client, model, force=args.force
    )
    build_collection("instA", DATA_DIR / "guidelines_instA.json", client, model, force=args.force)
    build_collection("instB", DATA_DIR / "guidelines_instB.json", client, model, force=args.force)

    # Schema erasure: same content as tau_old but with metadata stripped
    schema_erased_path = DATA_DIR / "guidelines_schema_erased.json"
    if not schema_erased_path.exists():
        tau_old_docs = json.loads((DATA_DIR / "guidelines_tau_old.json").read_text())
        erased = [{"id": g["id"], "text": g["text"]} for g in tau_old_docs]
        schema_erased_path.write_text(json.dumps(erased, indent=2))

    build_collection("schema_erased", schema_erased_path, client, model, force=args.force)

    print("Vector stores ready.")


if __name__ == "__main__":
    main()
