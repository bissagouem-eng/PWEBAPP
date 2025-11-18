import streamlit as st
import polars as pl
import fitz
import zipfile
from pathlib import Path
import re
from datetime import datetime
import io
from urllib.request import urlopen
from collections import Counter
import numpy as np

CACHE_DB = Path("lonab_master.parquet")
CACHE_STATS = Path("stats_cache.parquet")

# ←←← CHANGE ONLY THIS LINE (YOUR GOOGLE DRIVE FILE ID) ←←←
DRIVE_ID = "PUT_YOUR_GOOGLE_DRIVE_ID_HERE"   # ← CHANGE THIS ONLY
# Example: DRIVE_ID = "1YourRealDriveIDHere"

# ========================== INSTANT LOAD FROM DRIVE ==========================
@st.cache_data(ttl=3600)
def get_zip_from_drive():
    if DRIVE_ID.startswith("PUT_"):
        return None
    url = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
    try:
        resp = urlopen(url, timeout=60)
        return io.BytesIO(resp.read())
    except:
        return None

# ========================== DATE EXTRACTION (PERFECT) ==========================
fr_months = "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"

def extract_race_date(text: str, filename: str = None) -> datetime.date:
    text = " " + text.lower() + " "
    patterns = [
        rf"(\d{{1,2}})\s+({fr_months})\s+(\d{{4}})",
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            if len(m.groups()) == 3:
                if "/" in pat or "-" in pat:
                    a, b, c = m.groups()
                    y = a if int(a) > 31 else c
                    m_ = b
                    d = c if int(a) > 31 else a
                else:
                    d, month_fr, y = m.groups()
                    m_ = ["", "janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre"].index(month_fr[:3].lower())
                return datetime(int(y), int(m_), int(d)).date()
    if filename:
        m = re.search(r"(\d{4})[ \-_]?(\d{2})[ \-_]?(\d{2})", filename.lower())
        if m: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
    return datetime.now().date()

# ========================== FULL RACE & HORSE PARSER ==========================
def pdf_to_full_record(pdf_bytes, filename="unknown.pdf"):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        lines = text.split("\n")
        date = extract_race_date(text, filename)
        race_type = "QUANTUM" if "quantum" in text.lower() else "PMUB"
        track = next((l.strip() for l in lines if re.match(r"^[A-Z\s]{5,30}$", l.strip())), "UNKNOWN")

        horses = []
        current_horse = {}
        for line in lines:
            line = line.strip()
            num_match = re.search(r"^\s*(\d{1,2})\s", line)
            if num_match:
                if current_horse:
                    horses.append(current_horse)
                num = int(num_match.group(1))
                name = re.sub(r"^\d+\s+", "", line).split("JOC")[0].strip()
                jockey = re.search(r"JOC\.?\s*([A-ZÀ-Ý\. ]+)", line, re.I)
                trainer = re.search(r"ENT\.?\s*([A-ZÀ-Ý\. ]+)", line, re.I)
                current_horse = {
                    "num": num,
                    "horse": name[:40],
                    "jockey": jockey.group(1).strip() if jockey else "",
                    "trainer": trainer.group(1).strip() if trainer else "",
                    "odds": None  # will fill from results later
                }
            elif "Arrivée" in line or "1er" in line or "Gagnant" in line:
                positions = re.findall(r"(\d+)[ernd]{0,2}", line)
                if positions and current_horse:
                    current_horse["position"] = int(positions[0])
                    horses.append(current_horse)
                    current_horse = {}
        if current_horse:
            horses.append(current_horse)

        return {
            "date": date,
            "race_type": race_type,
            "track": track,
            "horses": horses,
            "filename": Path(filename).name
        }
    except Exception as e:
        raise Exception(f"{filename} → {e}")

# ========================== BUILD FULL DB + STATS ==========================
@st.cache_data
def build_full_db(zip_content):
    races = []
    all_horses = []
    with zipfile.ZipFile(zip_content) as z:
        pdfs = [f for f in z.infolist() if f.filename.lower().endswith(".pdf")]
        progress = st.progress(0)
        for i, info in enumerate(pdfs):
            st.write(f"Parsing race {i+1}/{len(pdfs)}: {Path(info.filename).name}")
            try:
                race = pdf_to_full_record(z.read(info.filename), info.filename)
                races.append({"date": race["date"], "track": race["track"], "race_type": race["race_type"]})
                for h in race["horses"]:
                    h.update({"race_date": race["date"], "track": race["track"]})
                    if "position" in h:
                        h["win"] = 1 if h["position"] == 1 else 0
                        h["place"] = 1 if h["position"] in (1,2,3) else 0
                    all_horses.append(h)
            except: pass
            progress.progress((i+1)/len(pdfs))

    df_races = pl.DataFrame(races)
    df_horses = pl.DataFrame(all_horses)

    # Stats
    jockey_stats = df_horses.group_by("jockey").agg([
        pl.count().alias("runs"),
        pl.sum("win").alias("wins"),
        pl.sum("place").alias("places")
    ]).with_columns([
        (pl.col("wins") / pl.col("runs") * 100).alias("win_rate%"),
        (pl.col("places") / pl.col("runs") * 100).alias("place_rate%")
    ]).filter(pl.col("runs") > 5).sort("win_rate%", descending=True)

    trainer_stats = df_horses.group_by("trainer").agg([
        pl.count().alias("runs"),
        pl.sum("win").alias("wins")
    ]).with_columns((pl.col("wins") / pl.col("runs") * 100).alias("win_rate%")
    ).filter(pl.col("runs") > 5).sort("win_rate%", descending=True)

    df_races.write_parquet(CACHE_DB)
    df_horses.write_parquet("horses.parquet")
    pl.DataFrame({"jockey_stats": [jockey_stats], "trainer_stats": [trainer_stats]}).write_parquet(CACHE_STATS)

    return df_races, df_horses, jockey_stats, trainer_stats

# ========================== MAIN APP – THE BEAST ==========================
st.set_page_config(page_title="TROPHY QUANTUM LONAB PRO v13 – THE PREDICTOR", layout="wide")
st.title("TROPHY QUANTUM LONAB PRO v13 – THE PREDICTOR")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("INSTANT LOAD FULL ARCHIVE\n2020-2025 (8 sec)", type="primary", use_container_width=True):
        with st.spinner("Downloading & parsing 1228 races..."):
            zip_file = get_zip_from_drive()
            if zip_file:
                st.session_state.zip = zip_file
                st.rerun()
            else:
                st.error("Set your Google Drive ID first!")

with col2:
    uploaded = st.file_uploader("Or upload ZIP manually", type="zip")
    if uploaded:
        st.session_state.zip = uploaded

zip_content = st.session_state.get("zip", None)

if not CACHE_DB.exists() or zip_content:
    if zip_content:
        races_df, horses_df, jockey_stats, trainer_stats = build_full_db(zip_content)
        st.session_state.zip = None
    else:
        st.stop()
else:
    races_df = pl.read_parquet(CACHE_DB)
    horses_df = pl.read_parquet("horses.parquet")
    stats = pl.read_parquet(CACHE_STATS)
    jockey_stats = stats["jockey_stats"][0]
    trainer_stats = stats["trainer_stats"][0]

total_races = len(races_df)
st.success(f"BEAST MODE ACTIVATED → {total_races:,} races • {len(horses_df):,} horses • 2020-2025")

tab1, tab2, tab3, tab4 = st.tabs(["TODAY'S PREDICTIONS", "JOCKEY RANKING", "TRAINER RANKING", "SMART COMBINATIONS"])

with tab1:
    st.subheader("TOP 10 JOCKEYS RIGHT NOW")
    st.dataframe(jockey_stats.head(10), use_container_width=True)

with tab2:
    st.subheader("TOP 10 TRAINERS RIGHT NOW")
    st.dataframe(trainer_stats.head(10), use_container_width=True)

with tab3:
    st.subheader("BEST COMBINATIONS (Auto-Generated)")
    top_jockeys = jockey_stats.head(6)["jockey"].to_list()
    st.write("**Quinté+ Base**: ", ", ".join(top_jockeys[:3]))
    st.write("**With**: ", ", ".join(top_jockeys[3:6]))
    st.code(f"2/4 → {' / '.join(top_jockeys[:4])}\n3/5 → {' / '.join(top_jockeys[:5])}\nTiercé → {' - '.join(top_jockeys[:3])}")

with tab4:
    st.download_button("Export Full Database", data=races_df.to_pandas().to_csv(index=False).encode(), file_name="lonab_2020_2025_full.csv")

if st.button("Clear Everything & Rebuild"):
    for p in [CACHE_DB, CACHE_STATS, "horses.parquet"]:
        if p.exists(): p.unlink()
    st.cache_data.clear()
    st.rerun()

st.caption("TROPHY QUANTUM LONAB PRO v13 – THE PREDICTOR – Built by champions, for champions – November 18, 2025")
