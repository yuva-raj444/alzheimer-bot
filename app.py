# app.py

```python
import streamlit as st
import faiss
import pickle
import numpy as np
import torch
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="EARACT AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0f1117;
        color: white;
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 10px;
    }

    .metric-card {
        background: #1e1e2f;
        padding: 15px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #2f2f45;
    }

    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4cc9f0;
    }

    .subtitle {
        font-size: 18px;
        color: #b0b3c6;
        margin-bottom: 30px;
    }

    .retrieval-box {
        background: #161b22;
        padding: 15px;
        border-radius: 14px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# HEADER
# =====================================================

st.markdown('<div class="title">🧠 EARACT AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Explainable Adaptive Retrieval-Augmented Conversational Transformer</div>',
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource

def load_generator_model():

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    return tokenizer, model


embedding_model = load_embedding_model()
tokenizer, generator_model = load_generator_model()

# =====================================================
# LOAD FILES
# =====================================================

@st.cache_resource

def load_index():
    return faiss.read_index("index.faiss")


@st.cache_resource

def load_metadata():
    with open("metadata.pkl", "rb") as f:
        return pickle.load(f)


index = load_index()
metadata = load_metadata()

# =====================================================
# ADAPTIVE RETRIEVAL
# =====================================================


def adaptive_k(query):

    words = len(query.split())

    if words <= 4:
        return 2

    elif words <= 10:
        return 4

    else:
        return 6

# =====================================================
# CONFIDENCE SCORE
# =====================================================


def calculate_confidence(distance):

    confidence = (1 / (1 + distance)) * 100

    confidence = max(0, min(100, confidence))

    return round(confidence, 2)

# =====================================================
# RETRIEVE CONTEXT
# =====================================================


def retrieve_context(query):

    top_k = adaptive_k(query)

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    retrieved = []

    for idx, dist in zip(indices[0], distances[0]):

        item = metadata[idx]

        retrieved.append({
            "question": item["question"],
            "answer": item["answer"],
            "distance": float(dist),
            "confidence": calculate_confidence(float(dist))
        })

    return retrieved

# =====================================================
# GENERATE ANSWER
# =====================================================


def generate_answer(query, retrieved_chunks):

    context = " ".join([
        item["answer"] for item in retrieved_chunks
    ])

    prompt = f"""
You are a helpful medical AI assistant.

Answer ONLY using the provided context.

If the answer is not available in context,
say:
"I could not find enough information."

You can answer greetings naturally.

Question: {query}

Context: {context}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = generator_model.generate(
        **inputs,
        max_length=120,
        num_beams=5,
        early_stopping=True,
        temperature=0.7
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, context

# =====================================================
# SEMANTIC SIMILARITY
# =====================================================


def semantic_similarity(query, answer):

    q_embed = embedding_model.encode([query])
    a_embed = embedding_model.encode([answer])

    similarity = cosine_similarity(q_embed, a_embed)[0][0]

    return round(float(similarity), 4)

# =====================================================
# BLEU SCORE
# =====================================================


def calculate_bleu(reference, generated):

    try:

        reference_tokens = [reference.split()]
        generated_tokens = generated.split()

        bleu = sentence_bleu(
            reference_tokens,
            generated_tokens
        )

        return round(float(bleu), 4)

    except:
        return 0.0

# =====================================================
# ROUGE SCORE
# =====================================================


def calculate_rouge(reference, generated):

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 4)
    }

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙ System Information")

    st.write("Embedding Model")
    st.info("all-MiniLM-L6-v2")

    st.write("Generator Model")
    st.info("t5-small")

    st.write("Vector Database")
    st.info("FAISS")

    st.write("Architecture")
    st.success("EARACT Framework")

# =====================================================
# CHAT MEMORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================================
# USER INPUT
# =====================================================

query = st.chat_input("Ask a medical question...")

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Generating Answer..."):

            retrieved = retrieve_context(query)

            final_answer, combined_context = generate_answer(
                query,
                retrieved
            )

            # =====================================
            # EVALUATIONS
            # =====================================

            top_distance = retrieved[0]["distance"]

            confidence = calculate_confidence(top_distance)

            similarity = semantic_similarity(
                query,
                final_answer
            )

            bleu = calculate_bleu(
                combined_context,
                final_answer
            )

            rouge = calculate_rouge(
                combined_context,
                final_answer
            )

            # =====================================
            # ANSWER DISPLAY
            # =====================================

            st.markdown(final_answer)

            # =====================================
            # METRICS
            # =====================================

            st.divider()

            st.subheader("📊 Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Confidence",
                    f"{confidence}%"
                )

            with col2:
                st.metric(
                    "Semantic Similarity",
                    similarity
                )

            with col3:
                st.metric(
                    "BLEU Score",
                    bleu
                )

            col4, col5 = st.columns(2)

            with col4:
                st.metric(
                    "ROUGE-1",
                    rouge["ROUGE-1"]
                )

            with col5:
                st.metric(
                    "ROUGE-L",
                    rouge["ROUGE-L"]
                )

            # =====================================
            # RETRIEVED CONTEXTS
            # =====================================

            st.divider()

            st.subheader("🔎 Retrieved Contexts")

            for i, item in enumerate(retrieved):

                with st.expander(f"Retrieved Context {i+1}"):

                    st.markdown("### Question")
                    st.info(item["question"])

                    st.markdown("### Answer")
                    st.write(item["answer"])

                    st.markdown("### Retrieval Metrics")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric(
                            "Distance",
                            round(item["distance"], 4)
                        )

                    with c2:
                        st.metric(
                            "Confidence",
                            f"{item['confidence']}%"
                        )

            # =====================================
            # COMBINED CONTEXT
            # =====================================

            with st.expander("📚 Combined Context"):
                st.write(combined_context)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "EARACT AI • Explainable Adaptive Retrieval-Augmented Conversational Transformer"
)
