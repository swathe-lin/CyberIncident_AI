
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CyberIncident AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/content")

INCIDENT_FILE = BASE_DIR / "data" / "processed" / "cleaned_incidents.csv"
GUIDANCE_FILE = BASE_DIR / "data" / "processed" / "cleaned_guidance.txt"

if not INCIDENT_FILE.exists():
    INCIDENT_FILE = BASE_DIR / "data" / "knowledge_base" / "incidents.csv"

if not GUIDANCE_FILE.exists():
    GUIDANCE_FILE = BASE_DIR / "data" / "knowledge_base" / "cybersecurity_guidance.txt"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f4f8fc;
        color: #12233f;
    }

    .main .block-container {
        max-width: 1500px;
        padding: 1.2rem 2rem 2rem 2rem;
    }


    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #08223f 0%,
            #0b3158 100%
        );
        min-width: 285px;
        max-width: 285px;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.1rem 1rem 1rem 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #edf6ff;
    }

    .sidebar-brand {
        padding: 6px 4px 18px 4px;
        border-bottom: 1px solid #244968;
        margin-bottom: 16px;
    }

    .sidebar-brand-title {
        font-size: 22px;
        font-weight: 800;
        line-height: 1.15;
    }

    .sidebar-brand-title span {
        color: #4fa3ff;
    }

    .sidebar-brand-subtitle {
        color: #9fc1df !important;
        font-size: 11px;
        margin-top: 5px;
    }

    .sidebar-status {
        background: #0b4a46;
        border: 1px solid #1cae92;
        border-radius: 9px;
        padding: 9px 10px;
        text-align: center;
        color: #63e6c2 !important;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .sidebar-section {
        color: #8fc8ff !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.2px;
        margin: 16px 0 8px 0;
    }

    .sidebar-card {
        background: #102d4d;
        border: 1px solid #214c73;
        border-radius: 9px;
        padding: 9px 10px;
        margin-bottom: 8px;
    }

    .sidebar-card-title {
        font-size: 12px;
        font-weight: 700;
    }

    .sidebar-card-value {
        color: #8fc8ff !important;
        font-size: 11px;
        margin-top: 3px;
        word-break: break-word;
    }

    .metric-card {
        background: #102d4d;
        border: 1px solid #214c73;
        border-radius: 9px;
        padding: 9px 10px;
        margin-bottom: 8px;
    }

    .metric-label {
        font-size: 11px;
        color: #d9e9f8 !important;
        font-weight: 700;
    }

    .metric-value {
        color: #55d6a3 !important;
        font-size: 19px;
        font-weight: 800;
        margin-top: 2px;
    }

    .sidebar-tagline {
        margin-top: 22px;
        padding: 15px 6px;
        text-align: center;
        border-top: 1px solid #244563;
        color: #9ab5d1 !important;
        font-size: 11px;
        font-style: italic;
        line-height: 1.6;
    }


    /* ================= RADIO ================= */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] {
        gap: 4px;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: 7px 8px;
        margin: 0;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {
        background: #123b62;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] p {
        font-size: 13px !important;
        font-weight: 600;
    }


    /* ================= HEADER ================= */

    .cyber-header {
        background: #ffffff;
        border: 1px solid #d9e5f1;
        border-radius: 16px;
        padding: 24px 30px 22px 30px;
        box-shadow: 0 7px 24px rgba(21, 65, 105, 0.08);
        margin-bottom: 22px;
    }

    .cyber-title {
        font-size: 38px;
        font-weight: 850;
        color: #101f3d;
        line-height: 1.08;
        margin: 0;
    }

    .cyber-title span {
        color: #166bea;
    }

    .cyber-subtitle {
        font-size: 17px;
        color: #27466c;
        margin-top: 8px;
        font-weight: 600;
    }

    .cyber-flow {
        font-size: 13px;
        color: #607895;
        margin-top: 13px;
    }


    /* ================= HEADINGS ================= */

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #102443;
        margin: 8px 0 4px 0;
    }

    .section-subtitle {
        font-size: 14px;
        color: #607895;
        margin-bottom: 13px;
    }


    /* ================= CARDS ================= */

    .result-card {
        background: #ffffff;
        border: 1px solid #d9e5f1;
        border-radius: 13px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(21, 65, 105, 0.06);
        margin-bottom: 14px;
    }

    .result-label {
        font-size: 12px;
        color: #6b809a;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    .result-value {
        font-size: 17px;
        color: #13294b;
        font-weight: 750;
        margin-top: 4px;
    }

    .badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        margin-top: 4px;
    }

    .badge-high {
        background: #ffe6e8;
        color: #c83243 !important;
    }

    .badge-medium {
        background: #fff2d7;
        color: #9b6500 !important;
    }

    .badge-critical {
        background: #ffd9de;
        color: #a91f32 !important;
    }

    .badge-low {
        background: #def7ec;
        color: #16734a !important;
    }

    .summary-card {
        background: #eefaf4;
        border: 1px solid #ccebdd;
        border-radius: 13px;
        padding: 19px;
        color: #214b3a;
        line-height: 1.65;
        font-size: 14px;
        min-height: 160px;
    }

    .summary-quote {
        color: #20ad75;
        font-size: 36px;
        font-weight: 900;
        line-height: .5;
        vertical-align: middle;
        margin-right: 7px;
    }

    .action-card {
        background: #ffffff;
        border: 1px solid #d9e5f1;
        border-radius: 11px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #243c5c;
    }

    .action-number {
        display: inline-block;
        width: 25px;
        height: 25px;
        line-height: 25px;
        text-align: center;
        border-radius: 50%;
        background: #1676e8;
        color: white !important;
        font-weight: 800;
        margin-right: 9px;
    }

    .knowledge-card {
        background: #ffffff;
        border: 1px solid #d9e5f1;
        border-radius: 12px;
        padding: 15px;
        min-height: 135px;
        box-shadow: 0 4px 14px rgba(21, 65, 105, 0.05);
    }

    .knowledge-title {
        font-size: 14px;
        font-weight: 800;
        color: #18385e;
        margin-bottom: 6px;
    }

    .knowledge-text {
        color: #68809c;
        font-size: 12px;
        line-height: 1.5;
    }


    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        border-top: 1px solid #d9e5f1;
        margin-top: 30px;
        padding: 18px 10px 5px 10px;
        color: #6c8199;
    }

    .footer-title {
        font-weight: 800;
        color: #25476b;
        font-size: 13px;
    }

    .footer-subtitle {
        font-size: 11px;
        margin-top: 3px;
    }

    .footer-tech {
        font-size: 11px;
        margin-top: 7px;
    }

    .footer-note {
        font-size: 10px;
        margin-top: 8px;
    }


    /* ================= STREAMLIT CONTROLS ================= */

    textarea {
        font-size: 14px !important;
        line-height: 1.55 !important;
    }

    .stButton > button {
        border-radius: 9px;
        min-height: 44px;
        font-size: 14px;
        font-weight: 750;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9e5f1;
        border-radius: 10px;
        padding: 10px;
    }


    @media (max-width: 900px) {
        .main .block-container {
            padding: 1rem;
        }

        .cyber-title {
            font-size: 30px;
        }

        .cyber-subtitle {
            font-size: 15px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_incidents(path):

    if not path.exists():

        return pd.DataFrame(
            columns=[
                "incident_id",
                "attack_type",
                "affected_system",
                "severity",
                "indicators",
                "description",
                "response_taken",
                "attack_techniques",
                "source_type"
            ]
        )

    df = pd.read_csv(path).fillna("")

    if "combined_text" not in df.columns:

        text_columns = [
            "attack_type",
            "affected_system",
            "severity",
            "indicators",
            "description",
            "response_taken",
            "attack_techniques"
        ]

        available_columns = [
            column for column in text_columns
            if column in df.columns
        ]

        df["combined_text"] = (
            df[available_columns]
            .astype(str)
            .agg(" ".join, axis=1)
        )

    return df


@st.cache_data
def load_guidance(path):

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


incidents_df = load_incidents(INCIDENT_FILE)
guidance_text = load_guidance(GUIDANCE_FILE)


# ============================================================
# TF-IDF
# ============================================================

@st.cache_resource
def build_tfidf(df):

    if df.empty:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        df["combined_text"]
        .fillna("")
        .astype(str)
    )

    return vectorizer, matrix


tfidf_vectorizer, tfidf_matrix = build_tfidf(
    incidents_df
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    try:

        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    except Exception:

        return None


embedding_model = load_embedding_model()


@st.cache_resource
def build_semantic_index(df):

    if (
        embedding_model is None
        or df.empty
    ):
        return None

    texts = (
        df["combined_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    return embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


incident_embeddings = build_semantic_index(
    incidents_df
)


# ============================================================
# SPACY NER
# ============================================================

@st.cache_resource
def load_spacy():

    try:

        import spacy

        return spacy.load(
            "en_core_web_sm"
        )

    except Exception:

        return None


nlp = load_spacy()


def extract_entities(text):

    entities = []

    if nlp is not None:

        doc = nlp(text)

        for ent in doc.ents:

            if ent.label_ in {
                "PERSON",
                "ORG",
                "GPE",
                "LOC"
            }:

                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_
                    }
                )


    lower_text = text.lower()


    # Cybersecurity attack types

    if not incidents_df.empty:

        attack_types = sorted(
            incidents_df["attack_type"]
            .astype(str)
            .unique()
            .tolist(),
            key=len,
            reverse=True
        )

        systems = sorted(
            incidents_df["affected_system"]
            .astype(str)
            .unique()
            .tolist(),
            key=len,
            reverse=True
        )

    else:

        attack_types = []
        systems = []


    for attack in attack_types:

        if (
            attack
            and attack.lower() in lower_text
        ):

            entities.append(
                {
                    "text": attack,
                    "label": "ATTACK_TYPE"
                }
            )


    for system in systems:

        if (
            system
            and system.lower() in lower_text
        ):

            entities.append(
                {
                    "text": system,
                    "label": "AFFECTED_SYSTEM"
                }
            )


    # Severity

    for severity in [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]:

        if re.search(
            r"\b"
            + re.escape(severity)
            + r"\b",
            text,
            re.I
        ):

            entities.append(
                {
                    "text": severity,
                    "label": "SEVERITY"
                }
            )


    # Remove duplicates

    unique = []
    seen = set()

    for item in entities:

        key = (
            item["text"].lower(),
            item["label"]
        )

        if key not in seen:

            seen.add(key)
            unique.append(item)

    return unique


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    top_k=5
):

    if (
        embedding_model is None
        or incident_embeddings is None
        or incidents_df.empty
    ):

        return pd.DataFrame()

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = np.dot(
        incident_embeddings,
        query_embedding[0]
    )

    order = np.argsort(scores)[::-1][:top_k]

    result = incidents_df.iloc[
        order
    ].copy()

    result["similarity"] = scores[order]

    return result.reset_index(
        drop=True
    )


# ============================================================
# TF-IDF SEARCH
# ============================================================

def tfidf_search(
    query,
    top_k=5
):

    if (
        tfidf_vectorizer is None
        or tfidf_matrix is None
        or incidents_df.empty
    ):

        return pd.DataFrame()

    query_vector = tfidf_vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).ravel()

    order = np.argsort(scores)[::-1][:top_k]

    result = incidents_df.iloc[
        order
    ].copy()

    result["tfidf_score"] = scores[order]

    return result.reset_index(
        drop=True
    )


# ============================================================
# GUIDANCE RETRIEVAL
# ============================================================

def retrieve_guidance(
    query,
    max_items=3
):

    if not guidance_text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            guidance_text
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        paragraphs
    )

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        matrix
    ).ravel()

    order = np.argsort(scores)[::-1][
        :max_items
    ]

    return [
        {
            "text": paragraphs[index],
            "score": float(
                scores[index]
            )
        }
        for index in order
    ]


# ============================================================
# INCIDENT ANALYSIS
# ============================================================

def infer_incident_fields(
    text,
    similar_df
):

    lower_text = text.lower()

    attack_type = "Security Incident"
    severity = "Medium"
    affected_system = "Unknown"


    # Attack type

    if not incidents_df.empty:

        for attack in incidents_df[
            "attack_type"
        ].astype(str).tolist():

            if (
                attack
                and attack.lower()
                in lower_text
            ):

                attack_type = attack
                break


    if (
        attack_type == "Security Incident"
        and not similar_df.empty
    ):

        attack_type = str(
            similar_df.iloc[0].get(
                "attack_type",
                attack_type
            )
        )


    # Severity

    for level in [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]:

        if re.search(
            r"\b"
            + re.escape(level)
            + r"\b",
            text,
            re.I
        ):

            severity = level
            break


    if (
        severity == "Medium"
        and not similar_df.empty
    ):

        severity = str(
            similar_df.iloc[0].get(
                "severity",
                severity
            )
        )


    # Affected system

    if not incidents_df.empty:

        for system in incidents_df[
            "affected_system"
        ].astype(str).tolist():

            if (
                system
                and system.lower()
                in lower_text
            ):

                affected_system = system
                break


    if (
        affected_system == "Unknown"
        and not similar_df.empty
    ):

        affected_system = str(
            similar_df.iloc[0].get(
                "affected_system",
                affected_system
            )
        )


    return (
        attack_type,
        severity,
        affected_system
    )


# ============================================================
# BADGE
# ============================================================

def badge_class(value):

    value = str(value).lower()

    if value == "critical":
        return "badge-critical"

    if value == "high":
        return "badge-high"

    if value == "medium":
        return "badge-medium"

    return "badge-low"


# ============================================================
# RESPONSE ACTIONS
# ============================================================

def build_actions(
    attack_type,
    severity
):

    attack = attack_type.lower()


    if "phishing" in attack:

        return [
            "Block the malicious URL, domain, and sender where applicable.",
            "Report and remove the suspicious email from affected mailboxes.",
            "Reset potentially exposed credentials and revoke active sessions.",
            "Review authentication activity for additional affected accounts.",
            "Preserve relevant email, endpoint, and authentication evidence."
        ]


    if "ransomware" in attack:

        return [
            "Isolate affected systems from the network.",
            "Preserve evidence and relevant system logs.",
            "Identify affected hosts, files, and accounts.",
            "Validate clean backups before restoration.",
            "Monitor restored systems for recurring suspicious activity."
        ]


    if "malware" in attack:

        return [
            "Isolate the affected endpoint or server.",
            "Preserve relevant endpoint and network evidence.",
            "Identify and remove the malicious software.",
            "Review connected systems for related indicators.",
            "Monitor systems after remediation."
        ]


    if (
        "denial" in attack
        or "ddos" in attack
    ):

        return [
            "Apply appropriate traffic filtering and rate limiting.",
            "Monitor traffic patterns and affected services.",
            "Identify the scope and duration of service impact.",
            "Review infrastructure logs for related indicators.",
            "Restore normal service and continue monitoring."
        ]


    return [
        "Validate the incident and determine its scope.",
        "Contain affected accounts, systems, or services as appropriate.",
        "Preserve relevant logs and other evidence.",
        "Review authentication, endpoint, network, and application activity.",
        "Recover affected services and continue monitoring."
    ]


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    attack_type,
    severity,
    affected_system
):

    summary = (
        f"A {severity.lower()}-severity "
        f"{attack_type.lower()} incident "
        f"was identified involving "
        f"{affected_system.lower()}. "
    )


    if "phishing" in attack_type.lower():

        summary += (
            "The reported activity indicates "
            "a suspicious message or fraudulent "
            "authentication workflow that may "
            "expose user credentials."
        )


    elif "ransomware" in attack_type.lower():

        summary += (
            "The reported activity is consistent "
            "with unauthorized encryption or "
            "disruption of files and requires "
            "rapid containment."
        )


    elif "malware" in attack_type.lower():

        summary += (
            "The report contains indicators "
            "associated with suspicious software "
            "execution and possible unauthorized "
            "system activity."
        )


    else:

        summary += (
            "The incident should be validated "
            "against available logs and historical "
            "cases before response decisions are "
            "finalized."
        )


    return summary


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-brand-title">
                🛡️ CyberIncident <span>AI</span>
            </div>

            <div class="sidebar-brand-subtitle">
                Smarter Insights. Safer Tomorrow.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-status">● AI SYSTEM READY</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-section">NAVIGATION</div>',
        unsafe_allow_html=True
    )


    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📄 Analyze Incident",
            "🕘 Incident History",
            "🗄️ Knowledge Base",
            "📊 Reports & Insights",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )


    st.markdown(
        '<div class="sidebar-section">KNOWLEDGE BASE</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                📁 Historical Incidents
            </div>

            <div class="sidebar-card-value">
                {len(incidents_df)} records
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                📚 RAG Guidance
            </div>

            <div class="sidebar-card-value">
                Curated response guidance
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                🧠 Embedding Model
            </div>

            <div class="sidebar-card-value">
                all-MiniLM-L6-v2
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                🔎 Semantic Retrieval
            </div>

            <div class="sidebar-card-value">
                Cosine similarity
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-section">MODEL EVALUATION</div>',
        unsafe_allow_html=True
    )


    evaluation_metrics = [
        ("🎯 Precision@5", "0.417"),
        ("🔄 Recall@5", "1.000"),
        ("📈 MRR@5", "0.900")
    ]


    for label, value in evaluation_metrics:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="sidebar-tagline">
            "Detect. Analyze.<br>
            Respond. Secure."
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="cyber-header">

        <div class="cyber-title">
            🛡️ CyberIncident <span>AI</span>
        </div>

        <div class="cyber-subtitle">
            Cybersecurity Incident Report Intelligence System
        </div>

        <div class="cyber-flow">
            Analyze incidents &nbsp;•&nbsp;
            Identify threat entities &nbsp;•&nbsp;
            Retrieve similar cases &nbsp;•&nbsp;
            Generate source-grounded recommendations
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="section-title">Welcome to CyberIncident AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            AI-assisted cybersecurity incident analysis using
            NER, TF-IDF, embeddings, semantic retrieval and RAG.
        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    Knowledge Base
                </div>

                <div class="result-value">
                    20 synthetic incident records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    Semantic Representation
                </div>

                <div class="result-value">
                    384-dimensional embeddings
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    Core Concepts
                </div>

                <div class="result-value">
                    NER • TF-IDF • Embeddings • RAG
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.info(
        "Use **Analyze Incident** from the sidebar "
        "to test the complete incident-analysis workflow."
    )


# ============================================================
# ANALYZE INCIDENT
# ============================================================

elif page == "📄 Analyze Incident":

    st.markdown(
        '<div class="section-title">📋 Incident Report</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Enter a cybersecurity incident description
            for AI-assisted analysis.
        </div>
        """,
        unsafe_allow_html=True
    )


    incident_text = st.text_area(
        "Incident report",
        value=(
            "An employee received a suspicious password reset "
            "email containing a fake login page and entered "
            "their credentials."
        ),
        height=150,
        label_visibility="collapsed",
        placeholder="Describe the cybersecurity incident here..."
    )


    analyze = st.button(
        "🔍  Analyze Incident",
        use_container_width=True,
        type="primary"
    )


    if analyze:

        if not incident_text.strip():

            st.warning(
                "Please enter an incident description "
                "before analysis."
            )

            st.stop()


        with st.spinner(
            "Analyzing incident and retrieving similar cases..."
        ):

            entities = extract_entities(
                incident_text
            )

            semantic_results = semantic_search(
                incident_text,
                top_k=5
            )

            tfidf_results = tfidf_search(
                incident_text,
                top_k=5
            )

            guidance_results = retrieve_guidance(
                incident_text,
                max_items=3
            )


            (
                attack_type,
                severity,
                affected_system
            ) = infer_incident_fields(
                incident_text,
                semantic_results
            )


            actions = build_actions(
                attack_type,
                severity
            )


            summary = build_summary(
                attack_type,
                severity,
                affected_system
            )


        st.markdown("---")


        st.markdown(
            '<div class="section-title">Incident Analysis Results</div>',
            unsafe_allow_html=True
        )


        left, right = st.columns(
            [1.05, 1.0],
            gap="large"
        )


        # ====================================================
        # LEFT
        # ====================================================

        with left:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Incident Type
                    </div>

                    <div class="result-value">
                        ⚠️ {attack_type}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Severity
                    </div>

                    <div class="result-value">

                        <span class="badge {badge_class(severity)}">
                            {severity}
                        </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Affected Area
                    </div>

                    <div class="result-value">
                        🛡️ {affected_system}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div class="result-card">

                    <div class="result-label">
                        Key Indicators / Entities
                    </div>
                """,
                unsafe_allow_html=True
            )


            if entities:

                for entity in entities:

                    st.markdown(
                        f"""
                        **{entity["label"]}**
                        — `{entity["text"]}`
                        """
                    )

            else:

                st.write(
                    "No named entities were extracted."
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Incident Summary
                    </div>

                    <div style="
                        color:#344e70;
                        font-size:13px;
                        line-height:1.6;
                        margin-top:7px;
                    ">
                        {summary}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # RIGHT
        # ====================================================

        with right:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        AI-Assisted Summary
                    </div>

                    <div class="summary-card">

                        <span class="summary-quote">
                            “
                        </span>

                        {summary}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div class="result-card">

                    <div class="result-label">
                        Recommended Actions
                    </div>
                """,
                unsafe_allow_html=True
            )


            for index, action in enumerate(
                actions,
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="action-card">

                        <span class="action-number">
                            {index}
                        </span>

                        {action}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # ====================================================
        # SIMILAR INCIDENTS
        # ====================================================

        st.markdown(
            '<div class="section-title">🔎 Similar Historical Incidents</div>',
            unsafe_allow_html=True
        )


        if not semantic_results.empty:

            display_columns = [
                column
                for column in [
                    "incident_id",
                    "attack_type",
                    "affected_system",
                    "severity",
                    "similarity"
                ]
                if column in semantic_results.columns
            ]


            display_df = semantic_results[
                display_columns
            ].copy()


            if "similarity" in display_df.columns:

                display_df["similarity"] = (
                    display_df["similarity"]
                    .round(3)
                )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Semantic retrieval is unavailable."
            )


        # ====================================================
        # RAG RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">📚 Related Knowledge (RAG Results)</div>',
            unsafe_allow_html=True
        )


        if guidance_results:

            knowledge_columns = st.columns(
                len(guidance_results)
            )


            for index, item in enumerate(
                guidance_results
            ):

                preview = (
                    item["text"]
                    .replace("\n", " ")
                    .strip()
                )


                if len(preview) > 220:

                    preview = (
                        preview[:220]
                        + "..."
                    )


                with knowledge_columns[index]:

                    st.markdown(
                        f"""
                        <div class="knowledge-card">

                            <div class="knowledge-title">
                                📄 Retrieved Guidance
                            </div>

                            <div class="knowledge-text">
                                {preview}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:

            st.info(
                "No guidance passages were retrieved."
            )


        # ====================================================
        # TF-IDF RESULTS
        # ====================================================

        with st.expander(
            "View TF-IDF retrieval results"
        ):

            if not tfidf_results.empty:

                columns = [
                    column
                    for column in [
                        "incident_id",
                        "attack_type",
                        "affected_system",
                        "severity",
                        "tfidf_score"
                    ]
                    if column in tfidf_results.columns
                ]


                temp = tfidf_results[
                    columns
                ].copy()


                if "tfidf_score" in temp.columns:

                    temp["tfidf_score"] = (
                        temp["tfidf_score"]
                        .round(3)
                    )


                st.dataframe(
                    temp,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# INCIDENT HISTORY
# ============================================================

elif page == "🕘 Incident History":

    st.markdown(
        '<div class="section-title">🕘 Incident History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Historical incident records available
            to the retrieval system.
        </div>
        """,
        unsafe_allow_html=True
    )


    if incidents_df.empty:

        st.warning(
            "No incident records were found."
        )

    else:

        st.dataframe(
            incidents_df.drop(
                columns=["combined_text"],
                errors="ignore"
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

elif page == "🗄️ Knowledge Base":

    st.markdown(
        '<div class="section-title">🗄️ Knowledge Base</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Sources used for incident retrieval
            and response guidance.
        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Historical Incident Dataset
                </div>

                <div class="result-value">
                    {len(incidents_df)}
                    synthetic demo records
                </div>

                <p style="
                    color:#607895;
                    font-size:13px;
                ">
                    Covers multiple cybersecurity incident
                    types, affected systems, severity levels,
                    indicators and response actions.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    Guidance Knowledge
                </div>

                <div class="result-value">
                    Curated defensive response guidance
                </div>

                <p style="
                    color:#607895;
                    font-size:13px;
                ">
                    Project guidance is organized for
                    retrieval during incident analysis.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with st.expander(
        "Preview guidance knowledge"
    ):

        st.text(
            guidance_text[:6000]
            if guidance_text
            else "Guidance file not found."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "📊 Reports & Insights":

    st.markdown(
        '<div class="section-title">📊 Reports & Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Overview of the current incident knowledge base.
        </div>
        """,
        unsafe_allow_html=True
    )


    if incidents_df.empty:

        st.warning(
            "No incident data available."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)


        with c1:
            st.metric(
                "Incidents",
                len(incidents_df)
            )


        with c2:
            st.metric(
                "Attack Types",
                incidents_df[
                    "attack_type"
                ].nunique()
            )


        with c3:
            st.metric(
                "Critical",
                int(
                    (
                        incidents_df["severity"]
                        == "Critical"
                    ).sum()
                )
            )


        with c4:
            st.metric(
                "High",
                int(
                    (
                        incidents_df["severity"]
                        == "High"
                    ).sum()
                )
            )


        st.markdown(
            "### Attack Type Distribution"
        )

        st.bar_chart(
            incidents_df[
                "attack_type"
            ].value_counts()
        )


        st.markdown(
            "### Severity Distribution"
        )

        st.bar_chart(
            incidents_df[
                "severity"
            ].value_counts()
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">ℹ️ About CyberIncident AI</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="result-card">

            <div class="result-value">
                Cybersecurity Incident Report
                Intelligence System
            </div>

            <p style="
                color:#607895;
                line-height:1.7;
                font-size:14px;
            ">
                CyberIncident AI is a defensive cybersecurity
                prototype designed to analyze incident descriptions,
                identify relevant entities, retrieve similar
                historical cases and surface source-grounded
                response guidance.
            </p>

            <p style="
                color:#607895;
                line-height:1.7;
                font-size:14px;
            ">
                Core concepts demonstrated:
                <b>
                NER, TF-IDF, embeddings,
                semantic retrieval, RAG and
                structured analysis.
                </b>
            </p>

            <p style="
                color:#607895;
                line-height:1.7;
                font-size:14px;
            ">
                The historical incident records in this prototype
                are synthetic demo data and should not be
                interpreted as real-world incident records.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <div class="footer-title">
            🛡️ CyberIncident AI
        </div>

        <div class="footer-subtitle">
            Cybersecurity Incident Report Intelligence System
        </div>

        <div class="footer-tech">
            NER • TF-IDF • Embeddings •
            Semantic Retrieval • RAG • Prompt Engineering
        </div>

        <div class="footer-note">
            AI-assisted analysis for defensive cybersecurity workflows.
            Always validate recommendations with qualified security personnel.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
