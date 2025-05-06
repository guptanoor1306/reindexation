import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Load secrets & init clients ──
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
        ids += [i["id"]["videoId"] for i in res["items"]]
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
        for v in resp["items"]:
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId": v["id"],
                "title":   v["snippet"]["title"],
                "thumb":   v["snippet"]["thumbnails"]["high"]["url"],
                "views":   int(v["statistics"].get("viewCount", 0)),
                "type":    "Short" if sec <= 180 else "Long-Form"
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    resp = openai_cli.embeddings.create(model="text-embedding-ada-002", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (np.linalg.norm(a)*np.linalg.norm(b))) * 100.0

def extract_text_via_vision(url: str) -> str:
    endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_KEY}"
    body = {"requests":[{"image":{"source":{"imageUri":url}},
                         "features":[{"type":"TEXT_DETECTION","maxResults":1}]}]}
    r = requests.post(endpoint, json=body).json()
    try:
        return r["responses"][0]["fullTextAnnotation"]["text"]
    except:
        return ""

# ── Sidebar ──
channel_id   = st.sidebar.text_input("Your Channel ID")
content_type = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤ 3 min)"])
num_matches  = st.sidebar.number_input("Results to show", 1, 10, 5)

if not channel_id:
    st.info("Enter your YouTube Channel ID in the sidebar.")
    st.stop()

# 1) Load & filter your uploads
with st.spinner("Loading your uploads…"):
    my_ids = fetch_my_videos(channel_id)
if not my_ids:
    st.error("No videos found for this channel.")
    st.stop()

df_all     = fetch_video_details(my_ids)
want_short = content_type.startswith("Shorts")
df         = df_all[df_all["type"] == ("Short" if want_short else "Long-Form")]
if df.empty:
    st.warning(f"No {content_type} found in your uploads.")
    st.stop()

# ── Main: select video & keyword ──
st.subheader("1) Select one of your videos")
sel_title = st.selectbox("Your videos", df["title"].tolist())
src       = df[df["title"] == sel_title].iloc[0]

# Show your selected video’s thumbnail & views
st.image(src["thumb"], caption=f"▶️ {sel_title}", width=300)
st.markdown(f"**Your Views:** {format_views(src['views'])}")

st.subheader("2) Enter a primary keyword (mandatory)")
primary_keyword = st.text_input("Primary keyword for broad matching")
if not primary_keyword:
    st.info("Please enter a primary keyword to proceed.")
    st.stop()

# Precompute source signals
emb_src  = get_embedding(src["title"])
text_src = extract_text_via_vision(src["thumb"])

# Precompute histogram of your thumbnail
src_img  = Image.open(requests.get(src["thumb"], stream=True).raw).convert("RGB").resize((256,256))
src_hist = src_img.histogram()
sum_src  = sum(src_hist)

def histogram_similarity(url: str) -> float:
    img  = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((256,256))
    hist = img.histogram()
    inter = sum(min(src_hist[i], hist[i]) for i in range(len(src_hist)))
    return (inter / sum_src) * 100.0

if st.button("3) Run Title & Thumbnail Match"):
    # Semantic title search
    sem = requests.get("https://www.googleapis.com/youtube/v3/search", params={
        "part":"snippet","q":src["title"],"type":"video",
        "order":"viewCount","maxResults":50,"key":YT_KEY
    }).json().get("items",[])
    cand_sem = [i["id"]["videoId"] for i in sem if i["snippet"]["channelId"] != channel_id]

    # Broad keyword search
    key = requests.get("https://www.googleapis.com/youtube/v3/search", params={
        "part":"snippet","q":primary_keyword,"type":"video",
        "order":"viewCount","maxResults":50,"key":YT_KEY
    }).json().get("items",[])
    cand_key = [i["id"]["videoId"] for i in key if i["snippet"]["channelId"] != channel_id]

    # Combine & fetch details
    combined_ids = list(dict.fromkeys(cand_sem + cand_key))
    if not combined_ids:
        st.warning("No candidates found.")
        st.stop()
    df_cand = fetch_video_details(combined_ids)

    # Title‐match scores
    df_cand["sem_sim"] = df_cand["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
    df_cand["key_sim"] = df_cand["title"].map(lambda t: fuzz.ratio(primary_keyword, t))
    df_cand["score"]   = df_cand[["sem_sim","key_sim"]].max(axis=1)
    df_cand.sort_values("score", ascending=False, inplace=True)

    # ── Table 1: Title Matches ──
    st.subheader("Table 1 – Title Matches")
    top_title = df_cand.head(num_matches)
    md1 = (
        "| Title | Type | Views | Sem % | Key % | Combined % |\n"
        "|:---   |:---  |:---:  |:---:  |:---:  |:---:  |\n"
    )
    for r in top_title.itertuples():
        link = f"https://youtu.be/{r.videoId}"
        md1 += (
            f"| [{r.title}]({link}) | {r.type} | {format_views(r.views)} | "
            f"{r.sem_sim:.1f}% | {r.key_sim:.1f}% | {r.score:.1f}% |\n"
        )
    st.markdown(md1, unsafe_allow_html=True)

    # ── Table 2: Thumbnail Matches ──
    st.subheader("Table 2 – Thumbnail Matches")
    df_thumb = df_cand.copy()
    df_thumb["text_sim"]   = df_thumb["thumb"].map(lambda u: fuzz.ratio(text_src, extract_text_via_vision(u)))
    df_thumb["hist_sim"]   = df_thumb["thumb"].map(histogram_similarity)

    # Keep only actual thumbnail matches
    df_thumb = df_thumb[(df_thumb["text_sim"] > 0) | (df_thumb["hist_sim"] > 0)]
    df_thumb.sort_values(["hist_sim","text_sim"], ascending=[False,False], inplace=True)
    top_thumb = df_thumb.head(num_matches)

    md2 = (
        "| Thumbnail | Title | Views | Text % | Visual % |\n"
        "|:---:      |:---   |---:   |---:    |---:     |\n"
    )
    for r in top_thumb.itertuples():
        link  = f"https://youtu.be/{r.videoId}"
        img   = f"![]({r.thumb})"
        md2  += (
            f"| {img} | [{r.title}]({link}) | {format_views(r.views)} | "
            f"{r.text_sim:.1f}% | {r.hist_sim:.1f}% |\n"
        )
    st.markdown(md2, unsafe_allow_html=True)
