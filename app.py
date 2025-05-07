import os
import re
import requests
import subprocess
import tempfile
import shutil
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from googleapiclient.discovery import build
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Only search within these 83 channels ──
ALLOWED_CHANNELS = [
    "UCK7tptUDHh-RYDsdxO1-5QQ","UCvJJ_dzjViJCoLf5uKUTwoA","UCvQECJukTDE2i6aCoMnS-Vg",
    # … add all other channel IDs …
    "UCczAxLCL79gHXKYaEc9k-ZQ","UCqykZoZjaOPb6i_Y5gk0kLQ"
]

# ── Secrets & clients ──
YT_KEY     = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
VISION_KEY = st.secrets["VISION"]["API_KEY"]

youtube    = build("youtube", "v3", developerKey=YT_KEY)
openai_cli = OpenAI(api_key=OPENAI_KEY)

# ── Helpers ──
def parse_iso_duration(dur: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    return (int(m.group(1) or 0)*3600 +
            int(m.group(2) or 0)*60 +
            int(m.group(3) or 0))

def format_views(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >=   1_000: return f"{n/1_000:.1f}K"
    return str(n)

@st.cache_data
def fetch_my_videos(cid: str) -> list[str]:
    ids, req = [], youtube.search().list(
        part="id", channelId=cid, type="video",
        order="date", maxResults=50
    )
    while req:
        res = req.execute()
        ids += [it["id"]["videoId"] for it in res.get("items", [])]
        req = youtube.search().list_next(req, res)
    return ids

@st.cache_data
def fetch_video_details(vids: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(vids), 50):
        chunk = vids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp.get("items", []):
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            pub = v["snippet"]["publishedAt"]
            rows.append({
                "videoId":    v["id"],
                "title":      v["snippet"]["title"],
                "type":       "Short" if sec <= 180 else "Long-Form",
                "channel":    v["snippet"]["channelTitle"],
                "uploadDate": datetime.fromisoformat(pub.rstrip("Z")).date().isoformat(),
                "thumb":      v["snippet"]["thumbnails"]["high"]["url"],
                "views":      int(v["statistics"].get("viewCount", 0))
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    resp = openai_cli.embeddings.create(model="text-embedding-ada-002", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (np.linalg.norm(a)*np.linalg.norm(b))) * 100.0

def extract_text_via_vision(url: str) -> str:
    r = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={VISION_KEY}",
        json={"requests":[{"image":{"source":{"imageUri":url}},
                           "features":[{"type":"TEXT_DETECTION","maxResults":1}]}]}
    ).json()
    try:
        return r["responses"][0]["fullTextAnnotation"]["text"]
    except:
        return ""

@st.cache_data(show_spinner=False)
def get_intro_text(video_id: str, seconds: int) -> str:
    tmpdir = tempfile.mkdtemp()
    try:
        out = os.path.join(tmpdir, f"{video_id}.webm")
        subprocess.run([
            "yt-dlp","-f","bestaudio",
            "--download-sections", f"*00:00:00-00:00:{seconds:02d}",
            "-o", out, f"https://youtu.be/{video_id}"
        ], check=True)
        with open(out,"rb") as f:
            resp = openai_cli.audio.transcriptions.create(model="whisper-1", file=f)
        return resp.text
    except subprocess.CalledProcessError:
        return ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── Sidebar ──
channel_id   = st.sidebar.text_input("Your Channel ID")
content_type = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤ 3 min)"])
num_matches  = st.sidebar.number_input("Results to show", 1, 10, 10)

if not channel_id:
    st.info("Enter your YouTube Channel ID."); st.stop()

with st.spinner("Loading your uploads…"):
    my_ids = fetch_my_videos(channel_id)
if not my_ids:
    st.error("No videos found."); st.stop()

df_all     = fetch_video_details(my_ids)
want_short = content_type.startswith("Shorts")
df         = df_all[df_all["type"] == ("Short" if want_short else "Long-Form")]
if df.empty:
    st.warning(f"No {content_type} found."); st.stop()

# ── 1) Select & display your video ──
st.subheader("1) Select one of your videos")
sel = st.selectbox("Your videos", df["title"].tolist())
src = df[df["title"] == sel].iloc[0]
st.image(src["thumb"], caption=f"▶️ {sel}", width=300)
st.markdown(
    f"**Channel:** {src['channel']}  "
    f"**Uploaded:** {src['uploadDate']}  "
    f"**Views:** {format_views(src['views'])}"
)

# ── 2) Enter a primary keyword ──
st.subheader("2) Enter a primary keyword (mandatory)")
pk = st.text_input("Primary keyword")
if not pk:
    st.info("Enter a primary keyword."); st.stop()

# ── Precompute embeddings & visuals ──
emb_src  = get_embedding(src["title"])
text_src = extract_text_via_vision(src["thumb"])
img      = Image.open(requests.get(src["thumb"], stream=True).raw)\
               .convert("RGB").resize((256,256))
hist_src = img.histogram(); total = sum(hist_src)
def hist_sim(url: str) -> float:
    img2 = Image.open(requests.get(url, stream=True).raw)\
                .convert("RGB").resize((256,256))
    h2   = img2.histogram()
    inter = sum(min(hist_src[i], h2[i]) for i in range(len(h2)))
    return inter/total*100.0

# ── 3) Run Title, Thumbnail & Intro Match ──
if st.button("3) Run Title, Thumbnail & Intro Match"):
    # search candidates
    def yt_search(q: str):
        return requests.get(
            "https://youtube.googleapis.com/youtube/v3/search",
            params=dict(part="snippet", q=q, type="video",
                        order="viewCount", maxResults=50, key=YT_KEY)
        ).json().get("items", [])
    sem_items = yt_search(src["title"])
    key_items = yt_search(pk)
    cand_sem  = [it["id"]["videoId"] for it in sem_items if it["snippet"]["channelId"] in ALLOWED_CHANNELS]
    cand_key  = [it["id"]["videoId"] for it in key_items if it["snippet"]["channelId"] in ALLOWED_CHANNELS]
    combined  = list(dict.fromkeys(cand_sem + cand_key))
    if not combined:
        st.warning("No matches found."); st.stop()

    df_cand = fetch_video_details(combined)

    # compute metrics
    df_cand["Sem %"]      = df_cand["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
    df_cand["Key %"]      = df_cand["title"].map(lambda t: fuzz.ratio(pk, t))
    df_cand["Combined %"] = df_cand[["Sem %","Key %"]].max(axis=1)
    df_cand["Text %"]     = df_cand["thumb"].map(lambda u: fuzz.ratio(text_src, extract_text_via_vision(u)))
    df_cand["Visual %"]   = df_cand["thumb"].map(hist_sim)

    secs  = 20 if want_short else 60
    intro = get_intro_text(src["videoId"], secs)
    if intro:
        df_cand["Intro→Title %"]     = df_cand["title"].map(lambda t: fuzz.ratio(intro, t))
        df_cand["Intro→ThumbText %"] = df_cand["thumb"].map(lambda u: fuzz.ratio(intro, extract_text_via_vision(u)))
        df_cand["Intro Combined %"]  = df_cand[["Intro→Title %","Intro→ThumbText %"]].max(axis=1)
    else:
        df_cand["Intro→Title %"]     = 0
        df_cand["Intro→ThumbText %"] = 0
        df_cand["Intro Combined %"]  = 0

    # ── Table 1 – Title Matches ──
    st.subheader("Table 1 – Title Matches")
    t1_long, t1_short = st.tabs(["Long-Form Matches","Shorts Matches"])
    for tab, vtype in [(t1_long, "Long-Form"), (t1_short, "Short")]:
        with tab:
            subset = df_cand[df_cand["type"] == vtype]
            topn   = subset.nlargest(num_matches, "Combined %")
            md = "| Title | Channel | Uploaded | Views | Sem % | Key % | Combined % |\n"
            md+= "| --- | --- | --- | ---: | ---: | ---: | ---: |\n"
            for _, r in topn.iterrows():
                md+= (
                    f"| [{r['title']}](https://youtu.be/{r.videoId}) | {r['channel']} | "
                    f"{r['uploadDate']} | {format_views(r['views'])} | "
                    f"{r['Sem %']:.1f}% | {r['Key %']:.1f}% | {r['Combined %']:.1f}% |\n"
                )
            st.markdown(md, unsafe_allow_html=True)

    # ── Table 2 – Thumbnail Matches ──
    st.subheader("Table 2 – Thumbnail Matches")
    t2_long, t2_short = st.tabs(["Long-Form Matches","Shorts Matches"])
    for tab, vtype in [(t2_long, "Long-Form"), (t2_short, "Short")]:
        with tab:
            subset = df_cand[
                (df_cand["type"] == vtype) &
                ((df_cand["Text %"] > 0) | (df_cand["Visual %"] > 0))
            ]
            topn = subset.sort_values(["Visual %","Text %"], ascending=[False,False])\
                         .head(num_matches)
            md = "| Thumbnail | Title | Channel | Uploaded | Views | Text % | Visual % |\n"
            md+= "| :---: | --- | --- | :---: | ---: | ---: | ---: |\n"
            for _, r in topn.iterrows():
                md+= (
                    f"| ![]({r['thumb']}) | [{r['title']}](https://youtu.be/{r.videoId}) | "
                    f"{r['channel']} | {r['uploadDate']} | {format_views(r['views'])} | "
                    f"{r['Text %']:.1f}% | {r['Visual %']:.1f}% |\n"
                )
            st.markdown(md, unsafe_allow_html=True)

    # ── Table 3 – Intro Text Matches ──
    st.subheader("Table 3 – Intro Text Matches")
    if not intro:
        st.warning("No audio transcript available.")
    else:
        t3_long, t3_short = st.tabs(["Long-Form Matches","Shorts Matches"])
        for tab, vtype in [(t3_long, "Long-Form"), (t3_short, "Short")]:
            with tab:
                subset = df_cand[df_cand["type"] == vtype]
                topn   = subset.nlargest(num_matches, "Intro Combined %")
                md = "| Title | Channel | Uploaded | Views | Intro→Title % | Intro→ThumbText % | Combined % |\n"
                md+= "| --- | --- | --- | ---: | ---: | ---: | ---: |\n"
                for _, r in topn.iterrows():
                    md+= (
                        f"| [{r['title']}](https://youtu.be/{r.videoId}) | {r['channel']} | "
                        f"{r['uploadDate']} | {format_views(r['views'])} | "
                        f"{r['Intro→Title %']:.1f}% | {r['Intro→ThumbText %']:.1f}% | {r['Intro Combined %']:.1f}% |\n"
                    )
                st.markdown(md, unsafe_allow_html=True)
