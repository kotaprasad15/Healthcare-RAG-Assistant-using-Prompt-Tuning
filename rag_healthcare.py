import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

# ---------- Paths ----------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "medical_faiss.index"       # make sure this matches your build_index script
META_PATH = "medical_metadata.pkl"       # list of {"question": ..., "answer": ...}

    
llama_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_index_and_data():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found at {META_PATH}")

    print("[*] Loading FAISS index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        qa_pairs = pickle.load(f)
    print(f"[*] Loaded {len(qa_pairs)} QA pairs from metadata.")
    return index, qa_pairs


def load_embedding_model():
    print("[*] Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def load_llama():
    print("[*] Loading Llama model:", llama_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # CPU Offload folder (use D:/ or E:/ for free space)
    offload_dir = "E:/HF_MODEL_OFFLOAD"
    os.makedirs(offload_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        llama_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # automatically uses GPU + CPU offload
        offload_folder=offload_dir,  # prevent OOM failures
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(llama_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device



def retrieve_context(query, index, qa_pairs, embed_model, k=5):
    # encode query to embedding
    query_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k)

    contexts = []
    for idx in indices[0]:
        qa = qa_pairs[idx]
        q = qa.get("question", "")
        a = qa.get("answer", "")
        contexts.append(f"Q: {q}\nA: {a}")

    context_text = "\n\n".join(contexts)

    # optionally truncate if extremely long
    if len(context_text) > 6000:
        context_text = context_text[:6000]

    return context_text


def build_prompt_llama(user_question, retrieved_context):
    system_instructions = (
        "You are a helpful medical information assistant.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "Do NOT make up facts.\n"
        "You are NOT a doctor and cannot give medical diagnosis or prescriptions.\n"
        "Always recommend consulting a qualified healthcare professional for personal medical issues.\n"
    )

    
    prompt = (
        "<s>[INST] "
        + system_instructions
        + "\n\n"
        + "=== CONTEXT START ===\n"
        + retrieved_context
        + "\n=== CONTEXT END ===\n\n"
        + "User question: "
        + user_question
        + "\n\n"
        + "Answer the question using ONLY the context above. "
          "If the answer is not in the context, clearly say you don't know based on the given information."
        + " [/INST]"
    )
    return prompt


def generate_answer_llama(prompt, tokenizer, model, device, max_new_tokens=512):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # deterministic
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Try to remove the prompt part and keep only the generated answer
    if prompt in full_text:
        answer = full_text.split(prompt, 1)[-1].strip()
    else:
        answer = full_text.strip()

    return answer


def main():
    index, qa_pairs = load_index_and_data()
    embed_model = load_embedding_model()
    tokenizer, llama_model, device = load_llama()

    print("\nHealthcare RAG Assistant (Llama 3B)")
    print("Note: This is NOT a real doctor. For serious issues, consult a professional.")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input("You: ")
        if user_q.lower().strip() in ["exit", "quit"]:
            break

        print("[*] Retrieving relevant context from medical knowledge base...")
        ctx = retrieve_context(user_q, index, qa_pairs, embed_model, k=5)

        print("[*] Building prompt and querying Llama 3B...")
        prompt = build_prompt_llama(user_q, ctx)

        answer = generate_answer_llama(prompt, tokenizer, llama_model, device)

        print("\nAssistant:", answer)
        print("-" * 80)


if __name__ == "__main__":
    main()
