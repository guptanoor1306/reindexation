import re, requests, streamlit as st, pandas as pd, numpy as np
from googleapiclient.discovery import build
from rapidfuzz import fuzz
from PIL import Image
import imagehash
import openai

st.set_page_config(layout="wide")
st.title("YouTube Semantically Related Finder")

# ── Secrets & Clients ──
YT_KEY    = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY= st.secrets["OPENAI"]["API_KEY"]
youtube   = build("youtube", "v3", developerKey=YT_KEY)
openai.api_key = OPENAI_KEY

# ── Utils ──
def parse_iso_duration(dur):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    h, mm, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h*3600 + mm*60 + s

@st.cache_data
def fetch_my_videos(ch_id):
    vids = []
    req = youtube.search().list(part="id", channelId=ch_id,
                                type="video", order="date", maxResults=50)
    while req:
        res = req.execute()
        vids += [i["id"]["videoId"] for i in res["items"]]
        req = youtube.search().list_next(req, res)
    return vids

@st.cache_data
def fetch_video_details(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            desc = v["snippet"]["description"] or ""
            sec  = parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId": v["id"],
                "title":   v["snippet"]["title"],
                "description": desc,
                "thumb":   v["snippet"]["thumbnails"]["high"]["url"],
                "views":   int(v["statistics"].get("viewCount", 0)),
                "type":    "Short" if sec <= 180 else "Long-Form"
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text):
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(resp["data"][0]["embedding"], dtype=np.float32)

def cosine_sim(a,b):
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))) * 100.0

@st.cache_data
def hash_image(url):
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return imagehash.phash(img)

# ── Sidebar Params ──
channel_id  = st.sidebar.text_input("Your Channel ID")
num_matches = st.sidebar.number_input("How many matches?", 1, 10, 5)
title_th    = st.sidebar.slider("Title ≥ (%)", 0, 100, 50)
thumb_th    = st.sidebar.slider("Max thumbnail hash dist", 0, 64, 10)
embed_th    = st.sidebar.slider("Embed ≥ (%)", 0, 100, 60)

# ── Main ──
if channel_id:
    with st.spinner("Loading your videos…"):
        my_ids = fetch_my_videos(channel_id)
        df     = fetch_video_details(my_ids)

    choice = st.selectbox("Pick one of your videos", df["title"])
    src     = df[df["title"]==choice].iloc[0]
    src_emb = get_embedding(src["description"])
    src_hash= hash_image(src["thumb"])

    if st.button("🔍 Find Semantically Similar"):
        # 1) Search YouTube by your description text
        search = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part":"snippet",
                "q": src["description"],
                "type":"video",
                "order":"viewCount",
                "maxResults": num_matches,
                "key": YT_KEY
            }
        ).json().get("items", [])
        cand_ids = [
            it["id"]["videoId"] for it in search
            if it["snippet"]["channelId"] != channel_id
        ]

        # 2) Fetch their details, embed & hash
        cdf = fetch_video_details(cand_ids)
        cdf["embedding"] = cdf["description"].map(get_embedding)
        cdf["sim"]       = cdf["embedding"].map(lambda e: cosine_sim(src_emb,e))
        cdf = cdf.sort_values("sim", ascending=False).head(num_matches)

        # 3) Render table
        md = "| Title | Type | Views | Title✅ | Thumb✅ | Embedding✅ |\n"
        md+= "|---|:---:|---:|:---:|:---:|:---:|\n"
        for _,r in cdf.iterrows():
            t_ok = "✅" if fuzz.ratio(src["title"],r["title"])>=title_th else "❌"
            h_ok = "✅" if (src_hash - hash_image(r["thumb"]))<=thumb_th else "❌"
            e_ok = "✅" if r["sim"]>=embed_th else "❌"
            link = f"https://youtube.com/watch?v={r['videoId']}"
            md += (
                f"| [{r['title']}]({link}) "
                f"| {r['type']} "
                f"| {r['views']:,} "
                f"| {t_ok} | {h_ok} | {e_ok} |\n"
            )
        st.markdown(md, unsafe_allow_html=True)
