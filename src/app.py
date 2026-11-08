"""
Streamlit Frontend for Production Hybrid RAG Application.
Provides user authentication, PDF document upload/management,
and real-time hybrid search chat with grounded citations and latency observability.
"""
import os
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide"
)

# Initialize Session State
if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

def get_auth_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def refresh_documents():
    """Fetch the authenticated user's uploaded documents."""
    if not st.session_state.token:
        return
    try:
        res = requests.get(f"{API_BASE_URL}/documents/", headers=get_auth_headers(), timeout=5)
        if res.status_code == 200:
            st.session_state.documents = res.json().get("documents", [])
    except Exception:
        pass

def attempt_login(email_val, pass_val):
    if not email_val or not pass_val:
        st.error("Please enter both email and password.")
        return
    try:
        res = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"email": email_val.strip(), "password": pass_val},
            timeout=8
        )
        if res.status_code == 200:
            data = res.json()
            st.session_state.token = data["access_token"]
            st.session_state.user_email = data["email"]
            refresh_documents()
            st.rerun()
        else:
            st.error("Invalid email or password. If you haven't created an account yet, click 'Create Account'.")
    except Exception as e:
        st.error(f"Cannot connect to FastAPI backend on {API_BASE_URL}. Ensure uvicorn is running: {e}")

def attempt_register(email_val, pass_val):
    if not email_val or not pass_val:
        st.error("Please enter both email and password.")
        return
    try:
        res = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"email": email_val.strip(), "password": pass_val},
            timeout=8
        )
        if res.status_code == 201:
            st.success("Account created successfully! Logging you in...")
            attempt_login(email_val, pass_val)
        elif res.status_code == 409:
            st.info("Account already exists! Logging in...")
            attempt_login(email_val, pass_val)
        else:
            detail = res.json().get("detail", "Registration failed.")
            st.error(f"Registration error: {detail}")
    except Exception as e:
        st.error(f"Cannot connect to FastAPI backend: {e}")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🛡️ Hybrid RAG")
    st.caption("Dense (MiniLM) + Sparse (BM42) + Qdrant RRF + CrossEncoder + LLM")

    if not st.session_state.token:
        st.subheader("🔐 Account Access")
        email = st.text_input("Email", key="auth_email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="auth_password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Log In", use_container_width=True, type="primary"):
                attempt_login(email, password)
        with col2:
            if st.button("Create Account", use_container_width=True):
                attempt_register(email, password)

        st.caption("Password requirements: 8+ characters, 1 uppercase, 1 lowercase, 1 digit.")

    else:
        # Authenticated Session View
        st.success(f"👤 **{st.session_state.user_email}**")
        if st.button("Log Out", use_container_width=True):
            st.session_state.token = None
            st.session_state.user_email = None
            st.session_state.messages = []
            st.session_state.documents = []
            st.rerun()

        st.divider()
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader("Upload PDF (Max 15MB, 150 pages)", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Index Document", use_container_width=True, type="primary"):
                with st.spinner("Validating, chunking, and computing embeddings..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        res = requests.post(
                            f"{API_BASE_URL}/documents/upload",
                            files=files,
                            headers=get_auth_headers(),
                            timeout=60
                        )
                        if res.status_code == 201:
                            st.success(f"Indexed {uploaded_file.name} successfully!")
                            refresh_documents()
                            st.rerun()
                        else:
                            st.error(res.json().get("detail", "Upload rejected."))
                    except Exception as e:
                        st.error(f"Upload error: {e}")

        st.divider()
        st.subheader("📚 Your Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{doc['original_filename']}**")
                    st.caption(f"{doc['chunk_count']} chunks | {round(doc['file_size_bytes']/1024, 1)} KB")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}"):
                        try:
                            dres = requests.delete(
                                f"{API_BASE_URL}/documents/{doc['id']}",
                                headers=get_auth_headers(),
                                timeout=10
                            )
                            if dres.status_code == 200:
                                st.toast(f"Deleted {doc['original_filename']}")
                                refresh_documents()
                                st.rerun()
                        except Exception as e:
                            st.error(f"Delete error: {e}")
        else:
            st.info("No documents uploaded yet.")

# ----------------- MAIN INTERFACE -----------------
st.title("Hybrid RAG")

if not st.session_state.token:
    st.info("👋 Welcome! Please log in or create an account in the sidebar to access your private documents and start querying.")
    st.stop()

# Query scope filter
doc_options = ["All Uploaded Documents"] + [d["original_filename"] for d in st.session_state.documents]
selected_doc = st.selectbox("Target Document Scope:", doc_options)
target_filename = None if selected_doc == "All Uploaded Documents" else selected_doc

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📑 Retrieved Evidence & Citations"):
                for cit in msg["citations"]:
                    st.markdown(
                        f"**Source:** `{cit['file_name']}` | **Score:** `{cit['relevance_score']}`\n\n"
                        f"> {cit['snippet']}"
                    )
        if "meta" in msg:
            st.caption(
                f"⏱️ `{msg['meta'].get('duration_ms', 0)}ms` | "
                f"⚡ `Cache: {'HIT' if msg['meta'].get('cache_hit') else 'MISS'}` | "
                f"🆔 `{msg['meta'].get('correlation_id', 'n/a')}`"
            )

# Chat Input
query = st.chat_input("Ask a question about your indexed documents...")
if query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Execute Hybrid RAG Query
    with st.chat_message("assistant"):
        with st.spinner("Executing hybrid retrieval, reranking, and generation..."):
            try:
                payload = {
                    "query": query,
                    "filename": target_filename,
                    "top_k": 5
                }
                res = requests.post(
                    f"{API_BASE_URL}/rag/query",
                    json=payload,
                    headers=get_auth_headers(),
                    timeout=45
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    meta = {
                        "duration_ms": data.get("duration_ms", 0),
                        "cache_hit": data.get("cache_hit", False),
                        "correlation_id": data.get("correlation_id", "n/a"),
                    }

                    # Prominently render the generated answer
                    st.markdown(answer)

                    if citations:
                        with st.expander("📑 Retrieved Evidence & Citations", expanded=False):
                            for cit in citations:
                                st.markdown(
                                    f"**Source Document:** `{cit['file_name']}` | **Passage Score:** `{cit['relevance_score']}`\n\n"
                                    f"> {cit['snippet']}\n"
                                )

                    st.caption(
                        f"⏱️ `{meta['duration_ms']}ms` | "
                        f"⚡ `Cache: {'HIT' if meta['cache_hit'] else 'MISS'}` | "
                        f"🆔 `{meta['correlation_id']}`"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "meta": meta
                    })

                elif res.status_code == 429:
                    retry_after = res.headers.get("Retry-After", "60")
                    st.error(f"⏳ Rate limit exceeded. Please wait {retry_after} seconds before querying again.")
                elif res.status_code == 400:
                    st.warning(f"⚠️ {res.json().get('detail', 'Query rejected by security guardrails.')}")
                else:
                    st.error(f"Error {res.status_code}: {res.json().get('detail', 'Unknown error')}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The upstream inference provider took too long to respond.")
            except Exception as e:
                st.error(f"Failed to communicate with RAG API: {e}")
