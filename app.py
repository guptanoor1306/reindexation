# app.py
import re, requests, streamlit as st, pandas as pd, numpy as np
from googleapiclient.discovery import build
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image
import imagehash

# OCR (optional—requires Tesseract installed on server)
try:
    import pytesseract
except ImportError:
    pytesseract = None

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Clients ──
YT_KEY     = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
youtube    = build("youtube", "v3", developerKey=YT_KEY)
client     = OpenAI(api_key=OPENAI_KEY)

# ── Helpers ──
def parse_iso_duration(dur: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    h, mm, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h*3600 + mm*60 + s

@st.cache_data
def fetch_my_videos(ch_id: str) -> list[str]:
    ids = []
    req = youtube.search().list(
        part="id", channelId=ch_id, type="video", order="date", maxResults=50
    )
    while req:
        res = req.execute()
        ids += [item["id"]["videoId"] for item in res["items"]]
        req = youtube.search().list_next(req, res)
    return ids

@st.cache_data
def fetch_video_details(video_ids: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(video_ids), 50):
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids[i:i+50])
        ).execute()
        for v in resp["items"]:
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId":     v["id"],
                "title":       v["snippet"]["title"],
                "description": v["snippet"].get("description","") or "",
                "thumb":       v["snippet"]["thumbnails"]["high"]["url"],
                "views":       int(v["statistics"].get("viewCount",0)),
                "type":        "Short" if sec <= 180 else "Long-Form"
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
    emb  = resp.data[0].embedding
    return np.array(emb, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b))) * 100.0

@st.cache_data
def hash_image(url: str) -> imagehash.ImageHash:
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return imagehash.phash(img)

# ── Sidebar ──
channel_id     = st.sidebar.text_input("Your Channel ID")
content_type   = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤3 min)"])
num_matches    = st.sidebar.number_input("How many results?", 1, 10, 5)
primary_keyword= st.sidebar.text_input("Primary keyword (fallback)")

# ── Main ──
if channel_id:
    with st.spinner("Loading your uploads…"):
        my_ids = fetch_my_videos(channel_id)
        df     = fetch_video_details(my_ids)

    # filter by duration
    want_short = content_type.startswith("Shorts")
    subset     = df[df["type"]=="Short"]=df["type"] if want_short else df[df["type"]=="Long-Form"]
    subset     = df[df["type"] == ("Short" if want_short else "Long-Form")]

    sel_title  = st.selectbox("Pick one of your videos", subset["title"])
    src        = subset[subset["title"]==sel_title].iloc[0]
    emb_src    = get_embedding(src["title"])
    thumb_src  = src["thumb"]
    hash_src   = hash_image(thumb_src)

    if st.button("Run Title & Thumbnail Match"):
        # ── Table 1: Title Matching ──
        # 1) Semantic retrieval (via YouTube keyword search + embed ranking)
        yt_q = src["title"]
        data = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part":"snippet",
                "q": yt_q,
                "type":"video",
                "order":"viewCount",
                "maxResults":50,
                "key":YT_KEY
            }
        ).json().get("items",[])
        cand = [i["id"]["videoId"] for i in data if i["snippet"]["channelId"]!=channel_id]
        details = fetch_video_details(cand)
        details["sim"] = details["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
        details = details.sort_values("sim", ascending=False)

        # 2) Fallback: if nothing or sim=0
        if details["sim"].max() == 0 and primary_keyword:
            data = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part":"snippet",
                    "q": primary_keyword,
                    "type":"video",
                    "order":"viewCount",
                    "maxResults":50,
                    "key":YT_KEY
                }
            ).json().get("items",[])
            cand = [i["id"]["videoId"] for i in data if i["snippet"]["channelId"]!=channel_id]
            details = fetch_video_details(cand)
            details["sim"] = details["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
            details = details.sort_values("sim", ascending=False)

        top1 = details.head(num_matches)

        st.subheader("Table 1 – Title Matches")
        md1 = "| Title | Type | Views | SemMatch % |\n|---|:---:|---:|---:|\n"
        for r in top1.itertuples():
            link = f"https://youtu.be/{r.videoId}"
            md1 += f"| [{r.title}]({link}) | {r.type} | {r.views:,} | {r.sim:.1f}% |\n"
        st.markdown(md1, unsafe_allow_html=True)

        # ── Table 2: Thumbnail Matching ──
        st.subheader("Table 2 – Thumbnail Matches")
        # OCR src
        if pytesseract:
            img_src = Image.open(requests.get(thumb_src,stream=True).raw)
            txt_src = pytesseract.image_to_string(img_src)
        else:
            txt_src = ""
            st.warning("Install Tesseract + pytesseract for thumbnail-text matching.")

        md2 = "| Title | TextMatch % | VisualMatch % |\n|---|---:|---:|\n"
        for r in top1.itertuples():
            # text
            if pytesseract:
                img2 = Image.open(requests.get(r.thumb,stream=True).raw)
                txt2 = pytesseract.image_to_string(img2)
                txt_sim = fuzz.ratio(txt_src, txt2)
            else:
                txt_sim = 0
            # visual
            hash2 = hash_image(r.thumb)
            dist  = hash_src - hash2
            vis_sim = max(0, (1 - dist/64)*100)
            link = f"https://youtu.be/{r.videoId}"
            md2 += f"| [{r.title}]({link}) | {txt_sim:.1f}% | {vis_sim:.1f}% |\n"
        st.markdown(md2, unsafe_allow_html=True)
