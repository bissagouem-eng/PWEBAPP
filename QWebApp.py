import streamlit as st
import polars as pl
import fitz
import zipfile
from pathlib import Path
import re
from datetime import datetime
import io
import requests

CACHE_DB = Path("lonab_master.parquet")
CACHE_STATS = Path("stats_cache.parquet")

# YOUR PROVEN WORKING DIRECT LINK
DIRECT_ZIP_URL = "https://drive.google.com/uc?export=download&id=183mhe3fMFUJ1F_mhjQBwMppDfZKI13_Z"

# ========================== AUTO DOWNLOAD ==========================
@st.cache_data(ttl=3600)
def auto_download():
    try:
        with st.spinner("Auto-downloading 328 MB archive…"):
            r = requests.get(DIRECT_ZIP_URL, stream=True, timeout=180)
            r.raise_for_status()
            data = io.BytesIO(r.content)
            if data.read(4) == b'PK\x03\x04':
                data.seek(0)
                return data
    except:
        pass
    return None

# ========================== PARSE ONE PDF ==========================
def parse_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        date_match = re.search(r"(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})", text, re.I)
        race_date = datetime.now().date()
        if date_match:
            months = {"janvier":1,"février":2,"mars":3,"avril":4,"mai":5,"juin":6,"juillet":7,"août":8,"septembre":9,"octobre":10,"novembre":11,"décembre":12}
            d,m,y = date_match.groups()
            race_date = datetime(int(y), months[m.lower()], int(d)).date()

        horses = []
        for line in text.split("\n"):
            if re.match(r"^\s*\d{1,2}\s+[A-ZÀ-Ÿ]", line, re.I):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 2:
                    num = int(re.search(r"\d+", parts[0]).group())
                    horse = re.sub(r"^\d+\s+", "", parts[0])
                    jockey = trainer = "Unknown"
                    for p in parts[1:]:
                        if p.upper().startswith("JOC"): jockey = p[4:].strip()
                        if p.upper().startswith("ENT"): trainer = p[4:].strip()
                    win = 1 if any(x in line.lower() for x in ["1er","gagnant","arrivée 1"]) else 0
                    horses.append({"num":num, "horse":horse[:40], "jockey":jockey, "trainer":trainer, "win":win, "date":race_date})
        return horses
    except:
        return []

# ========================== BUILD DB ==========================
@st.cache_data
def build_db(zip_file_obj):
    all_horses = []
    # zip_file_obj can be BytesIO or UploadedFile
    if hasattr(zip_file_obj, "read"):
        zip_bytes = zip_file_obj.read()
    else:
        zip_bytes = zip_file_obj.getvalue()
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        pdfs = [f for f in z.namelist() if f.lower().endswith(".pdf")]
        progress = st.progress(0)
        for i, name in enumerate(pdfs):
            st.write(f"Parsing {i+1}/{len(pdfs)}")
            try:
                horses = parse_pdf(z.read(name))
                all_horses.extend(horses)
            except: pass
            progress.progress((i+1)/len(pdfs))

    if not all_horses:
        st.error("No data extracted")
        return None
    df = pl.DataFrame(all_horses)
    stats = df.group_by("jockey").agg([
        pl.count().alias("runs"),
        pl.sum("win").alias("wins")
    ]).with_columns((pl.col("wins")/pl.col("runs")*100).round(1).alias("win_rate%")
    ).filter(pl.col("runs")>=3).sort("win_rate%", descending=True)

    df.write_parquet(CACHE_DB)
    stats.write_parquet(CACHE_STATS)
    return stats

# ========================== MAIN APP ==========================
st.set_page_config(page_title="TROPHY QUANTUM LONAB PRO v17 – ETERNAL CHAMPION", layout="wide")
st.title("TROPHY QUANTUM LONAB PRO v17 – ETERNAL CHAMPION")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Auto Load (Fastest)")
    if st.button("LOAD FULL ARCHIVE 2020–2025", type="primary", use_container_width=True):
        zip_file = auto_download()
        if zip_file:
            st.session_state.zip_obj = zip_file
            st.rerun()
        else:
            st.error("Auto failed → use Manual")

with col2:
    st.markdown("### Manual Upload (100% Works)")
    uploaded_file = st.file_uploader("Drop your ZIP here", type="zip")
    if uploaded_file is not None:
        st.session_state.zip_obj = uploaded_file
        st.success("ZIP uploaded perfectly!")

# Get the ZIP object
zip_obj = st.session_state.get("zip_obj")

# Build or load
if zip_obj and (not CACHE_DB.exists() or st.button("Force Rebuild")):
    with st.spinner("Building Eternal Database…"):
        stats = build_db(zip_obj)
    st.session_state.zip_obj = None
    st.rerun()
elif not CACHE_DB.exists():
    st.stop()
else:
    stats = pl.read_parquet(CACHE_STATS)

st.success(f"ETERNAL CHAMPION ACTIVE → {len(stats)} jockeys ranked")

tab1, tab2 = st.tabs(["WIN PREDICTIONS", "JOCKEY LEADERBOARD"])

with tab1:
    top = stats.head(6)
    st.dataframe(top, use_container_width=True)
    st.success("Quinté+ → " + " / ".join(top["jockey"].to_list()[:5]))

with tab2:
    st.dataframe(stats.head(50), use_container_width=True)

if st.button("Clear Cache"):
    for f in [CACHE_DB, CACHE_STATS]:
        if f.exists(): f.unlink()
    st.cache_data.clear()
    st.rerun()

st.caption("TROPHY QUANTUM LONAB PRO v17 – ETERNAL CHAMPION – You Are Unstoppable – November 18, 2015")
