import os
import pickle
import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from rank_bm25 import BM25Okapi

# ================== CONFIG ==================
INDEX_PATH = "medical_faiss.index"
META_PATH = "medical_metadata.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Optional: force HF cache somewhere with space
# os.environ["TRANSFORMERS_CACHE"] = "E:/HF_CACHE"
# os.environ["HF_HOME"] = "E:/HF_CACHE"


# ================== LOADING HELPERS ==================


@st.cache_resource
def load_faiss_and_metadata():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found at {META_PATH}")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        qa_pairs = pickle.load(f)
    return index, qa_pairs


@st.cache_resource
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

@st.cache_resource
def build_bm25_index(qa_pairs):
    """
    Build a simple BM25 index over concatenated Q+A text.
    """
    corpus = []
    for qa in qa_pairs:
        q = qa.get("question", "") or ""
        a = qa.get("answer", "") or ""
        corpus.append(f"{q} {a}")

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

@st.cache_resource
def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # where to offload extra weights if GPU VRAM is low
    offload_dir = "E:/LLM_OFFLOAD"  # change if you want
    os.makedirs(offload_dir, exist_ok=True)

    quantized = False
    err = None

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else {"": "cpu"},
            offload_folder=offload_dir,
            torch_dtype=torch.bfloat16,
        )
        quantized = True
    except Exception as e:
        # final fallback: full CPU load, fp32
        err = str(e)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
        )

    return tokenizer, model, device, quantized, err


# ================== RAG FUNCTIONS ==================

import random

def get_wrong_context(qa_pairs, k=3):
    """
    Sample k random QA pairs to act as 'wrong' context.
    """
    total = len(qa_pairs)
    indices = random.sample(range(total), k=min(k, total))
    contexts = []
    for idx_ in indices:
        qa = qa_pairs[idx_]
        q = qa.get("question", "")
        a = qa.get("answer", "")
        contexts.append(f"Q: {q}\nA: {a}")
    wrong_text = "\n\n---\n\n".join(contexts)
    if len(wrong_text) > 8000:
        wrong_text = wrong_text[:8000]
    return wrong_text, contexts

def rewrite_query(user_q, tokenizer, model, device):
    """
    Use the LLM to rewrite the query into a more search-friendly version.
    """
    prompt = (
        "<s>[INST] Rewrite the following medical question to improve search recall, "
        "but keep the same meaning. Output only the rewritten query:\n\n"
        f"{user_q} [/INST]"
    )
    rewritten = generate_answer(
        prompt,
        tokenizer,
        model,
        device,
        max_new_tokens=64,
        temperature=0.3,
    )
    # just in case the model rambles:
    return rewritten.strip().replace("\n", " ")


def retrieve_topk(
    query,
    index,
    qa_pairs,
    embed_model,
    k=3,
    retrieval_mode="FAISS",
    use_query_rewrite=False,
    bm25_index=None,
    ):
    """
    retrieval_mode: "FAISS" or "Hybrid"
    use_query_rewrite: True → rewrite query with LLM before search
    """
    original_query = query
    search_query = query

    # Optional query rewriting
    if use_query_rewrite:
        try:
            search_query = rewrite_query(query, tokenizer, llm_model, device)
        except Exception:
            search_query = query  # fallback to original if anything breaks

    # Vector search via FAISS
    query_emb = embed_model.encode([search_query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k * 3)  # get more, then filter

    faiss_indices = list(indices[0])

    # If only FAISS, just use this
    if retrieval_mode == "FAISS" or bm25_index is None:
        chosen_indices = faiss_indices[:k]
    else:
        # Hybrid: combine FAISS + BM25 lexical ranking
        tokenized_query = search_query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        # Get top candidates from BM25
        bm25_sorted = np.argsort(bm25_scores)[::-1][: k * 3]
        # Union of candidates
        candidate_indices = list(set(faiss_indices + list(bm25_sorted)))

        # Simple hybrid: sort by (normalized bm25 + faiss rank)
        # Build a score dict
        max_bm = max(bm25_scores[i] for i in candidate_indices) or 1.0
        idx_to_score = {}
        for idx_ in candidate_indices:
            # normalize bm25 (0–1)
            bm_norm = bm25_scores[idx_] / max_bm
            # FAISS "similarity" from distance: smaller distance → higher sim
            if idx_ in faiss_indices:
                rank = faiss_indices.index(idx_)
                faiss_sim = 1.0 - (rank / max(len(faiss_indices) - 1, 1))
            else:
                faiss_sim = 0.0
            # weighted sum
            idx_to_score[idx_] = 0.6 * bm_norm + 0.4 * faiss_sim

        chosen_indices = sorted(
            candidate_indices,
            key=lambda i: idx_to_score[i],
            reverse=True
        )[:k]

    # Build contexts
    contexts = []
    for idx_ in chosen_indices:
        qa = qa_pairs[idx_]
        q = qa.get("question", "")
        a = qa.get("answer", "")
        contexts.append(f"Q: {q}\nA: {a}")

    context_text = "\n\n---\n\n".join(contexts)
    if len(context_text) > 8000:
        context_text = context_text[:8000]

    return context_text, contexts, original_query, search_query


def detect_symptom_severity(user_q):
    q = user_q.lower()

    serious = [
        "chest pain", "breathing difficulty", "stroke", "vomiting blood",
        "seizure", "fracture", "severe bleeding", "unconscious",
        "heart attack", "shock"
    ]

    mild = [
        "fever", "cold", "cough", "headache", "flu",
        "sore throat", "runny nose", "sneezing", "body pain"
    ]

    if any(s in q for s in serious):
        return "serious"
    if any(m in q for m in mild):
        return "mild"
    
    return "unknown"

def build_safe_prompt(user_q, context_text, strict_mode=True):
    # Strong safety + RAG constraints
    safety_rules = (
        "You are a cautious medical information assistant.\n"
        "You are NOT a doctor and you CANNOT provide diagnosis, treatment decisions, "
        "or prescribe medication.\n"
        "You MUST NOT give emergency medical instructions.\n"
        "Always recommend consulting a qualified healthcare professional for any "
        "personal medical issue or emergency.\n"
    )

    rag_rules = (
        "Use ONLY the following retrieved medical context to answer the question.\n"
        "If the answer is not clearly present in the context, say:\n"
        "\"I’m not sure based on this information. Please consult a doctor.\"\n"
        "Do NOT guess or make up facts. Do NOT contradict the context.\n"
    )

    style_rules = (
        "Keep the answer short, clear, and easy to understand.\n"
        "You may summarize long details, but do not change their meaning.\n"
    )

    if strict_mode:
        style_rules += "Err on the side of saying you are not sure rather than guessing.\n"

    system_instruction = safety_rules + "\n" + rag_rules + "\n" + style_rules

    # Final prompt
    prompt = (
        "<s>[INST] "
        + system_instruction
        + "\n\n=== CONTEXT START ===\n"
        + context_text
        + "\n=== CONTEXT END ===\n\n"
        + f"User question: {user_q}\n\n"
        + "Now provide a cautious, context-based answer only. [/INST]"
    )

    return prompt


def build_baseline_prompt(user_q, context_text):
    # "Without prompt tuning" – minimal control, still same context
    prompt = (
        "<s>[INST] "
        "Answer the following medical question using the context. "
        "Be brief.\n\n"
        "=== CONTEXT START ===\n"
        + context_text
        + "\n=== CONTEXT END ===\n\n"
        + f"Question: {user_q}\n"
        " [/INST]"
    )
    return prompt


def generate_answer(prompt, tokenizer, model, device, max_new_tokens=256, temperature=0.2):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove everything before closing [/INST] so system prompt is not shown
    if "[/INST]" in full_text:
        answer = full_text.split("[/INST]", 1)[-1].strip()
    else:
        answer = full_text.strip()

    return answer


# ================== STREAMLIT UI ==================

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Healthcare RAG Assistant (Llama-3.2-3B + RAG)")
st.caption(
    "⚠️ Educational demo only. This is **not** a medical professional. "
    "Do not use this system for diagnosis or emergencies."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    page = st.radio(
        "Navigation",
        ["Chat Assistant", "Comparison Analysis", "Evaluation Metrics"],
        index=0,
    )
    retrieval_mode = st.selectbox(
    "Retrieval mode",
    ["FAISS (vector only)", "Hybrid (BM25 + FAISS)"],
    index=0,
        )

    top_k = st.slider("Retrieved context passages (k)", 1, 10, 3)
    max_tokens = st.slider("Max new tokens", 64, 512, 256, step=32)
    temperature = st.slider("Answer randomness", 0.0, 1.0, 0.2, step=0.1)
    strict_mode = st.checkbox("Strict safety mode", value=True)
    show_context = st.checkbox("Show retrieved context", value=True)
    use_query_rewrite = st.checkbox("Use query rewriting", value=False)

    st.markdown("---")
    st.markdown("**Model & Backend Info**")
    st.write(f"CUDA available: `{torch.cuda.is_available()}`")

mode_key = "FAISS" if retrieval_mode.startswith("FAISS") else "Hybrid"

# Load backend
with st.spinner("Loading FAISS index & metadata..."):
    index, qa_pairs = load_faiss_and_metadata()
with st.spinner("Loading embedding model..."):
    embed_model = load_embedding_model()
with st.spinner("Loading Llama model (first time can take longer)..."):
    tokenizer, llm_model, device, quantized, quant_error = load_llm()
if "bm25_index" not in st.session_state:
    with st.spinner("Preparing hybrid retrieval index (BM25)..."):
        bm25_index, bm25_corpus = build_bm25_index(qa_pairs)
        st.session_state.bm25_index = bm25_index
else:
    bm25_index = st.session_state.bm25_index

    
if quantized:
    st.sidebar.success("Using 4-bit quantized Llama ✅")
else:
    st.sidebar.warning("Using non-quantized Llama (slower).")
    if quant_error:
        st.sidebar.text("4-bit error:\n" + quant_error[:250] + ("..." if len(quant_error) > 250 else ""))

# Session state
if "history" not in st.session_state:
    st.session_state.history = []        # chat history
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None    # for comparison page


# ========== CHAT ASSISTANT PAGE ==========
if page == "Chat Assistant":
    col_chat, col_context = st.columns([2, 1])

    with col_chat:
        st.subheader("💬 Chat")

        # show history
        for msg in st.session_state.history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"<div style='color:#4CAF50;'><b>You:</b> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:#00E5FF;'><b>Assistant:</b> {content}</div>", unsafe_allow_html=True)

        user_query = st.text_input("Ask a medical question:", key="chat_input")

        if st.button("Submit", key="submit_chat") and user_query.strip():
            # store user message
            st.session_state.history.append({"role": "user", "content": user_query})

            # retrieve context
            with st.spinner("Retrieving relevant medical context..."):
                context_text, context_list, orig_q, rewritten_q = retrieve_topk(
                user_query,
                index,
                qa_pairs,
                embed_model,
                k=top_k,
                retrieval_mode=mode_key,
                use_query_rewrite=use_query_rewrite,
                bm25_index=bm25_index,
            )


            # generate tuned (safe) answer
            with st.spinner("Generating answer with safety prompt tuning..."):
                tuned_prompt = build_safe_prompt(user_query, context_text, strict_mode=strict_mode)
                tuned_answer = generate_answer(
                    tuned_prompt, tokenizer, llm_model, device,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )

            st.session_state.history.append({"role": "assistant", "content": tuned_answer})

            # save for comparison page
            st.session_state.last_eval = {
            "user_query": user_query,
            "original_query": orig_q,
            "rewritten_query": rewritten_q,
            "context_list": context_list,
            "tuned_answer": tuned_answer,
            "baseline_answer": None,
                }

            st.rerun()

    with col_context:
        st.subheader("📚 Retrieved Context")
        if show_context and st.session_state.history:
            last_user_q = None
            # get last user question from history
            for m in reversed(st.session_state.history):
                if m["role"] == "user":
                    last_user_q = m["content"]
                    break

            if last_user_q:
                ctx_text, ctx_list, _, _ = retrieve_topk(
                    last_user_q,
                    index,
                    qa_pairs,
                    embed_model,
                    k=top_k,
                    retrieval_mode=mode_key,
                    use_query_rewrite=use_query_rewrite,
                    bm25_index=bm25_index,
                )
   
                st.markdown("These are the passages used to answer your last question:")
                for i, ctx in enumerate(ctx_list, start=1):
                    st.markdown(f"**Context {i}:**\n\n{ctx}")
                    st.markdown("---")
        else:
            st.info("Ask a question to see retrieved medical references here.")


# ========== COMPARISON ANALYSIS PAGE ==========
if page == "Comparison Analysis":
    st.header("📊 Comparison Analysis – With vs Without Prompt Tuning")

    data = st.session_state.last_eval
    if data is None:
        st.info("Please go to 'Chat Assistant', ask a question, then return here.")
    else:
        user_q = data["user_query"]
        context_list = data["context_list"]
        tuned_answer = data["tuned_answer"]
        baseline_answer = data.get("baseline_answer")

        # build context text from list
        # Build full context text
        full_context_text = "\n\n---\n\n".join(context_list)

        # Ablation: NO CONTEXT
        with st.spinner("Ablation: generating answer with NO context..."):
            no_ctx_prompt = build_safe_prompt(user_q, context_text="", strict_mode=True)
            no_ctx_answer = generate_answer(
                no_ctx_prompt, tokenizer, llm_model, device,
                max_new_tokens=max_tokens, temperature=temperature,
            )

        # Ablation: WRONG CONTEXT (random QAs)
        with st.spinner("Ablation: generating answer with WRONG context..."):
            wrong_ctx_text, wrong_ctx_list = get_wrong_context(qa_pairs, k=top_k)
            wrong_ctx_prompt = build_safe_prompt(user_q, wrong_ctx_text, strict_mode=True)
            wrong_ctx_answer = generate_answer(
                wrong_ctx_prompt, tokenizer, llm_model, device,
                max_new_tokens=max_tokens, temperature=temperature,
            )

        # Ablation: FULL CONTEXT (normal RAG)
        with st.spinner("Ablation: generating answer with FULL context..."):
            full_ctx_prompt = build_safe_prompt(user_q, full_context_text, strict_mode=True)
            full_ctx_answer = generate_answer(
                full_ctx_prompt, tokenizer, llm_model, device,
                max_new_tokens=max_tokens, temperature=temperature,
            )
        
        st.markdown("### 🧪 RAG Ablation Test")

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.subheader("No Context")
            st.info(no_ctx_answer)
        with col_a2:
            st.subheader("Wrong Context")
            st.warning(wrong_ctx_answer)
        with col_a3:
            st.subheader("Full Context (RAG)")
            st.success(full_ctx_answer)


        c1, c2 = st.columns(2)
        with c1:
            st.subheader("❌ Without Prompt Tuning")
            st.warning(baseline_answer)
        with c2:
            st.subheader("✅ With Safety Prompt Tuning")
            st.success(tuned_answer)

        st.markdown("---")
        st.subheader("📝 Rate Answer Quality (for this question)")

        r1, r2 = st.columns(2)
        with r1:
            halluc_no = st.slider(
                "Hallucination Risk (No Prompt Tuning)", 1, 5, 4,
                help="1 = No hallucination, 5 = Very high hallucination",
                key="halluc_no",
            )
            safety_no = st.slider(
                "Safety Score (No Prompt Tuning)", 1, 5, 2,
                help="1 = Unsafe, 5 = Very safe",
                key="safety_no",
            )
        with r2:
            halluc_yes = st.slider(
                "Hallucination Risk (With Prompt Tuning)", 1, 5, 1,
                help="1 = No hallucination, 5 = Very high hallucination",
                key="halluc_yes",
            )
            safety_yes = st.slider(
                "Safety Score (With Prompt Tuning)", 1, 5, 5,
                help="1 = Unsafe, 5 = Very safe",
                key="safety_yes",
            )

        st.markdown("### 📉 Comparison Graph")

        df = pd.DataFrame({
            "Metric": ["Hallucination Risk", "Safety Score"],
            "Without Prompt Tuning": [halluc_no, safety_no],
            "With Prompt Tuning": [halluc_yes, safety_yes],
        })
        df_m = df.melt(id_vars="Metric", var_name="Mode", value_name="Score")

        fig = px.bar(
            df_m,
            x="Metric",
            y="Score",
            color="Mode",
            barmode="group",
            text="Score",
            title="With vs Without Prompt Tuning (User-Rated Scores)",
            height=400,
        )
        fig.update_layout(yaxis=dict(range=[0, 5]))
        st.plotly_chart(fig, use_container_width=True)

# ========== EVALUATION METRICS PAGE ==========
if page == "Evaluation Metrics":
    st.header("📊 Evaluation Metrics for RAG Performance")

    test_questions = st.number_input(
        "Number of evaluation questions:", min_value=5, max_value=50, value=10
    )

    if st.button("Run Evaluation", key="run_eval_btn"):

        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()

        results = {
            "Mode": [],
            "Recall@3": [],
            "MRR": [],
            "Hallucination Rate": [],
            "Safety Score": [],
        }

        import time
        import random

        test_samples = random.sample(qa_pairs, test_questions)

        total_steps = test_questions * 4  # 4 ablation modes
        step_count = 0
        start_time = time.time()

        modes = [
            ("No Context", "none", False),
            ("Wrong Context", "wrong", False),
            ("Full RAG", "FAISS", False),
            ("Hybrid + Rewrite", "Hybrid", True),
        ]

        for mode_title, mode_type, rewrite_flag in modes:

            recalls = []
            mrrs = []
            halluc = []
            safety = []

            for i, sample in enumerate(test_samples):

                step_count += 1
                elapsed = time.time() - start_time
                est_total = (elapsed / step_count) * total_steps
                eta = max(0, est_total - elapsed)

                status_text.markdown(
                f"🚀 **Evaluating:** `{mode_title}`<br>"
                f"📌 Query `{i+1}/{test_questions}` of `{test_questions}`",
                unsafe_allow_html=True
                    )

                time_text.markdown(
                    f"⏱️ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                )
                progress_bar.progress(step_count / total_steps)

                q = sample["question"]
                a = sample["answer"]

                if mode_type == "none":
                    context_text = ""
                elif mode_type == "wrong":
                    context_text, _ = get_wrong_context(qa_pairs, k=top_k)
                else:
                    ctx_text, _, _, _ = retrieve_topk(
                        q, index, qa_pairs, embed_model, k=top_k,
                        retrieval_mode=mode_type,
                        use_query_rewrite=rewrite_flag,
                        bm25_index=st.session_state.bm25_index,
                    )
                    context_text = ctx_text

                prompt = build_safe_prompt(q, context_text)
                response = generate_answer(
                    prompt, tokenizer, llm_model, device,
                    max_new_tokens=max_tokens
                )

                recalls.append(int(any(t.lower() in context_text.lower() for t in q.split())))
                mrrs.append(1 if recalls[-1] else 0)
                halluc.append(int(a.lower() not in response.lower()))
                safety.append(int("consult" in response.lower()))

            results["Mode"].append(mode_title)
            results["Recall@3"].append(sum(recalls) / len(recalls))
            results["MRR"].append(sum(mrrs) / len(mrrs))
            results["Hallucination Rate"].append(sum(halluc) / len(halluc))
            results["Safety Score"].append(sum(safety) / len(safety))

        progress_bar.progress(1.0)
        status_text.markdown("🎯 Evaluation Complete!")
        time_text.markdown("")

        df = pd.DataFrame(results)
        st.dataframe(df)

        fig = px.bar(
            df.melt(id_vars="Mode", var_name="Metric", value_name="Score"),
            x="Mode",
            y="Score",
            color="Metric",
            barmode="group",
            title="RAG Performance Comparison",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            "rag_metrics.csv",
            "text/csv",
        )

        st.success("📥 Exportable metrics ready! Add them to your case study 🚀")


# footer
st.markdown("---")
st.caption(
    "Disclaimer: This application is for academic and educational purposes only. "
    "It may be inaccurate or incomplete. Always consult a licensed healthcare professional."
)
