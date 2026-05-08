import streamlit as st
import faiss
import pickle
import numpy as np

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

    :root {
        color-scheme: light dark;
        --app-bg: #ffffff;
        --app-fg: #111827;
        --app-accent: #2563eb;
        --app-muted: #6b7280;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #0f1117;
            --app-fg: #ffffff;
            --app-accent: #4cc9f0;
            --app-muted: #b0b3c6;
        }
    }

    .stApp {
        background-color: var(--app-bg);
        color: var(--app-fg);
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: var(--app-accent);
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: var(--app-muted);
        margin-bottom: 25px;
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 10px;
    }

    #MainMenu,
    header,
    footer,
    [data-testid="stToolbar"],
    [data-testid="stHeader"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"] {
        visibility: hidden;
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">EARACT AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Explainable Adaptive Retrieval-Augmented Conversational Transformer</div>',
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODELS
# =====================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
GENERATOR_MODEL = "google/flan-t5-base"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_generator_model():

    tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL)

    model = T5ForConditionalGeneration.from_pretrained(
        GENERATOR_MODEL
    )

    return tokenizer, model


embedding_model = load_embedding_model()

tokenizer, generator_model = load_generator_model()

# =====================================================
# LOAD SAVED FILES
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

    confidence = max(0, 100 - (distance * 10))

    return round(confidence, 2)

# =====================================================
# RETRIEVE CONTEXT
# =====================================================

def retrieve_context(query):

    top_k = adaptive_k(query)

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved = []

    for idx, dist in zip(indices[0], distances[0]):

        item = metadata[idx]

        retrieved.append({
             "question": item["Questions"],
             "answer": item["Answers"],
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
You are EARACT AI, a helpful medical assistant specialized in Alzheimer’s Disease.

Use the provided context to answer the user's question clearly and naturally.

Provide detailed but concise answers.
Reply for Hi,Hello,Hey greetings.
If the answer is not found in the context, say:
"I could not find enough information."

Question:
{query}

Context:
{context}

Detailed Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = generator_model.generate(
    **inputs,
    max_new_tokens=220,
    min_length=50,
    do_sample=True,
    temperature=0.85,
    top_p=0.92,
    repetition_penalty=1.3
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

    similarity = cosine_similarity(
        q_embed,
        a_embed
    )[0][0]

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
    st.info(EMBEDDING_MODEL)

    st.write("Generator Model")
    st.info(GENERATOR_MODEL)

    st.write("Vector Database")
    st.info("FAISS")

    st.write("Architecture")
    st.success("EARACT Framework")

# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================================
# CHAT INPUT
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

            confidence = max(0, 100 - (top_distance * 10))

            confidence = round(confidence, 2)

            reference_answer = retrieved[0]["answer"]

            similarity = semantic_similarity(
                reference_answer,
                final_answer
            )

            bleu = calculate_bleu(
                reference_answer,
                final_answer
            )

            rouge = calculate_rouge(
                reference_answer,
                final_answer
            )

            # =====================================
            # ANSWER
            # =====================================

            st.markdown(final_answer)

            # =====================================
            # METRICS
            # =====================================

            st.divider()

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Confidence",
                f"{confidence}%"
            )

            col2.metric(
                "Semantic Similarity",
                similarity
            )

            col3.metric(
                "BLEU Score",
                bleu
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "ROUGE-1",
                rouge["ROUGE-1"]
            )

            col5.metric(
                "ROUGE-L",
                rouge["ROUGE-L"]
            )

            # =====================================
            # RETRIEVED CONTEXTS
            # =====================================

            st.divider()

            st.subheader("Retrieved Contexts")

            for i, item in enumerate(retrieved):

                with st.expander(f"Retrieved Context {i+1}"):

                    st.markdown("### Question")
                    st.info(item["question"])

                    st.markdown("### Answer")
                    st.write(item["answer"])

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Distance",
                        round(item["distance"], 4)
                    )

                    c2.metric(
                        "Confidence",
                        f"{item['confidence']}%"
                    )

            # =====================================
            # COMBINED CONTEXT
            # =====================================

            with st.expander("Combined Context"):
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
