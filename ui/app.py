import os, requests, time
import streamlit as st
from io import BytesIO
from PIL import Image

API_URL = os.getenv("API_URL", "http://search-api:8000")
TOP_K_DEFAULT = int(os.getenv("TOP_K", 10))
MODEL     = os.getenv("MODEL",     "RN50")
ES_INDEX  = os.getenv("ES_INDEX",  "images")

# ───────────────────────── sidebar “About” panel ─────────────────────────
with st.sidebar.expander("🔧 Runtime info", expanded=False):
    try:
        meta = requests.get(f"{API_URL}/meta", timeout=5).json()
    except Exception as e:
        st.error(f"Meta fetch failed: {e}")
    else:
        st.markdown(
            f"""
            **Model**: `{meta['model_name']}`  
            **Dim**: `{meta['vector_dim']}`  
            **Device**: `{meta['device']}`  
            **ES index**: `{meta['es_index']}`  
            **Docs**: `{meta['doc_count']}`  
            **ES v**: `{meta['es_version']}` ({meta['cluster']})  
            **HNSW**: m={meta['hnsw_m']} ef={meta['hnsw_ef']}
            """
        )

# ───────────────────────── main UI (search form) ─────────────────────────

st.title("🔍 Image Search Demo")

with st.form(key="search_form"):
    query = st.text_input("Enter a prompt:", "sunset over mountains")
    k = st.slider("How many images", 1, 20, TOP_K_DEFAULT)

    # this button both shows on-screen and is the “submit” for the form
    submitted = st.form_submit_button("Search")

if submitted and query:
    t0 = time.time()
    with st.spinner("Querying…"):
        resp = requests.post(
            f"{API_URL}/search/text",
            json={"text": query, "k": k},
            timeout=20,
        )
    if resp.ok:
        hits = resp.json()
        cols = st.columns(4)
        for i, h in enumerate(hits):
            col = cols[i % 4]
            img_resp = requests.get(f"{API_URL}{h['url']}", timeout=20)
            img = Image.open(BytesIO(img_resp.content))
            col.image(img, caption=f"{h['score']:.3f}")
    else:
        st.error(f"Error {resp.status_code}: {resp.text}")
    latency_ms = (time.time() - t0) * 1000
    st.write(f"⏱️ {latency_ms:.1f} ms")
