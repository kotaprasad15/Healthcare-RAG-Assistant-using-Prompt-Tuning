import numpy as np
import faiss
from tqdm import tqdm
import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os
import pandas as pd
import os
import torch

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "medical_faiss.index"
META_PATH = "medical_metadata.pkl"


def load_medquad():
    path = r"E:\Text_Analytics_Project\medquad.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print("Loading MedQuAD CSV dataset...")

    df = pd.read_csv(path, encoding="utf-8")

    # Required columns: question, answer
    qa = []
    for _, r in df.iterrows():
        q = r.get("question")
        a = r.get("answer")
        if isinstance(q, str) and isinstance(a, str) and len(q.strip()) > 0 and len(a.strip()) > 0:
            qa.append({"question": q.strip(), "answer": a.strip()})

    # Remove duplicates
    qa = list({(item['question'], item['answer']): item for item in qa}.values())

    print("MedQuAD:", len(qa), "entries")
    return qa

def load_doctor_healthcare():
    path = r"E:\Text_Analytics_Project\HealthCareMagic-100k.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print("Loading Doctor-Healthcare-100k dataset...")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa = []
    for r in data:
        q = r.get("Patient")
        a = r.get("Doctor")
        if isinstance(q, str) and isinstance(a, str) and len(q.strip()) > 0 and len(a.strip()) > 0:
            qa.append({"question": q.strip(), "answer": a.strip()})

    print("Doctor-Healthcare:", len(qa), "entries")
    return qa


def load_general_medical_qna():
    path = r"E:\Text_Analytics_Project\en_medical_dialog.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print("Loading General Medical Dialog dataset...")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa = []
    for r in data:
        q = r.get("Patient") or r.get("question")
        a = r.get("Doctor") or r.get("answer")
        if isinstance(q, str) and isinstance(a, str) and len(q.strip()) > 0 and len(a.strip()) > 0:
            qa.append({"question": q.strip(), "answer": a.strip()})

    print("General Medical Dialog:", len(qa), "entries")
    return qa

def build_index():
    # Load datasets
    all_qa = []
    all_qa.extend(load_medquad())
    all_qa.extend(load_doctor_healthcare())
    all_qa.extend(load_general_medical_qna())

    print("\nTotal QA Pairs:", len(all_qa))

    # Embed all text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    texts = [f"Q: {x['question']}\nA: {x['answer']}" for x in all_qa]

    embeddings = []
    print("\nEmbedding documents...")
    for i in tqdm(range(0, len(texts), 64)):
        batch = texts[i:i+64]
        batch_emb = embed_model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_emb)

    embeddings = np.vstack(embeddings).astype("float32")

    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Index size:", index.ntotal)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(all_qa, f)

    print("\n[✔] Combined medical knowledge base created successfully!")

if __name__ == "__main__":
    build_index()
