import re
import json
import sqlite3
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi

# ── SQLite cache setup ──
conn = sqlite3.connect("cache.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS search_cache (
  channel_id TEXT,
  query TEXT,
  video_ids TEXT,
  fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(channel_id, query)
)
""")
conn.commit()

def get_cached_search(channel_id: str, query: str):
    cur.execute(
        "SELECT video_ids FROM search_cache WHERE channel_id=? AND query=?",
        (channel_id, query)
    )
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

def set_cached_search(channel_id: str, query: str, video_ids: list[str]):
    js = json.dumps(video_ids)
    cur.execute(
        "INSERT OR REPLACE INTO search_cache(channel_id,query,video_ids) VALUES (?,?,?)",
        (channel_id, query, js)
    )
    conn.commit()

def global_search(query: str, max_results: int = 50) -> list[str]:
    """Global YouTube search, cached."""
    cached = get_cached_search("", query)
    if cached is not None:
        return cached
    resp = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        order="viewCount",
        maxResults=max_results
    ).execute()
    vids = [item["id"]["videoId"] for item in resp.get("items", [])]
    set_cached_search("", query, vids)
    return vids

# ── Streamlit & API clients ──
st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

ALLOWED_CHANNELS = [
    # … all 83 channel IDs …
    "UCK7tptUDHh-RYDsdxO1-5QQ","UCvJJ_dzjViJCoLf5uKUTwoA","UCvQECJukTDE2i6aCoMnS-Vg",
    # (snip for brevity) …
    "UCczAxLCL79gHXKYaEc9k-ZQ","UCqykZoZjaOPb6i_Y5gk0kLQ"
]

YT_KEY     = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
VISION_KEY = st.secrets["VISION"]["API_KEY"]

youtube    = build("youtube", "v3", developerKey=YT_KEY)
openai_cli = OpenAI(api_key=OPENAI_KEY)

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
    """Fetch your own uploads via the uploads playlist (low quota cost)."""
    try:
        ch = youtube.channels().list(part="contentDetails", id=cid).execute()
        uploads_pl = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as e:
        st.error(f"Error fetching your uploads playlist: {e}")
        return []
    vids = []
    req = youtube.playlistItems().list(
        part="contentDetails", playlistId=uploads_pl, maxResults=50
    )
    while req:
        res = req.execute()
        vids += [item["contentDetails"]["videoId"] for item in res.get("items", [])]
        req = youtube.playlistItems().list_next(req, res)
    return vids

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
                "channelId":  v["snippet"]["channelId"],
                "channel":    v["snippet"]["channelTitle"],
                "uploadDate": datetime.fromisoformat(pub.rstrip("Z")).date().isoformat(),
                "thumb":      v["snippet"]["thumbnails"]["high"]["url"],
                "views":      int(v["statistics"].get("viewCount", 0)),
                "type":       "Short" if sec <= 180 else "Long-Form"
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

def get_intro_text(video_id: str, seconds: int) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text, total = "", 0
        for seg in transcript:
            if total >= seconds:
                break
            text += " " + seg["text"]
            total += seg["duration"]
        return text.strip()
    except:
        return ""

# ── Sidebar ──
channel_id   = st.sidebar.text_input("Your Channel ID")
content_type = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤ 3 min)"])
num_matches  = st.sidebar.number_input("Results to show", 1, 10, 5)

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

# ── Select & display your source video ──
st.subheader("1) Select one of your videos")
sel = st.selectbox("Your videos", df["title"].tolist())
src = df[df["title"] == sel].iloc[0]

st.image(src["thumb"], caption=f"▶️ {sel}", width=300)
st.markdown(
    f"**Channel:** {src['channel']}   "
    f"**Uploaded:** {src['uploadDate']}   "
    f"**Views:** {format_views(src['views'])}"
)

st.subheader("2) Enter a primary keyword (mandatory)")
pk = st.text_input("Primary keyword")
if not pk:
    st.info("Enter a primary keyword."); st.stop()

# ── Precompute embeddings & thumbnail histogram ──
emb_src  = get_embedding(src["title"])
text_src = extract_text_via_vision(src["thumb"])
img      = Image.open(requests.get(src["thumb"], stream=True).raw)\
               .convert("RGB").resize((256,256))
hist_src = img.histogram(); total = sum(hist_src)
def hist_sim(url: str) -> float:
    i = Image.open(requests.get(url, stream=True).raw)\
             .convert("RGB").resize((256,256))
    h = i.histogram()
    return sum(min(hist_src[j], h[j]) for j in range(len(h))) / total * 100

# ── Run all three matches ──
if st.button("3) Run Title, Thumbnail & Intro Match"):
    # global semantic + keyword
    sem_ids = global_search(src["title"])
    key_ids = global_search(pk)
    combined = list(dict.fromkeys(sem_ids + key_ids))
    if not combined:
        st.warning("No matches found."); st.stop()

    # fetch details + filter to your 83 channels
    df_cand = fetch_video_details(combined)
    df_cand = df_cand[df_cand["channelId"].isin(ALLOWED_CHANNELS)]
    if df_cand.empty:
        st.warning("No matches found within your channel list."); st.stop()

    # Table 1 – Title Matches
    df_cand["Sem %"]      = df_cand["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
    df_cand["Keyword %"]  = df_cand["title"].map(lambda t: fuzz.ratio(pk, t))
    df_cand["Combined %"] = df_cand[["Sem %","Keyword %"]].max(axis=1)
    df_cand.sort_values("Combined %", ascending=False, inplace=True)

    st.subheader("Table 1 – Title Matches")
    md1 = "| Title | Channel | Uploaded | Views | Sem % | Keyword % | Combined % |\n"
    md1 += "|---|---|---|---:|---:|---:|---:|\n"
    for r in df_cand.head(num_matches).itertuples():
        url = f"https://youtu.be/{r.videoId}"
        md1 += (
            f"| [{r.title}]({url}) | {r.channel} | {r.uploadDate} | "
            f"{format_views(r.views)} | {r._9:.1f}% | {r._10:.1f}% | {r._11:.1f}% |\n"
        )
    st.markdown(md1, unsafe_allow_html=True)

    # Table 2 – Thumbnail Matches
    df_cand["Text %"]   = df_cand["thumb"].map(lambda u: fuzz.ratio(text_src, extract_text_via_vision(u)))
    df_cand["Visual %"] = df_cand["thumb"].map(hist_sim)
    df2 = df_cand[(df_cand["Text %"]>0)|(df_cand["Visual %"]>0)]
    df2.sort_values(["Visual %","Text %"], ascending=[False,False], inplace=True)

    st.subheader("Table 2 – Thumbnail Matches")
    md2 = "| Thumbnail | Title | Channel | Uploaded | Views | Text % | Visual % |\n"
    md2 += "|:---:|---|---|:---:|---:|---:|---:|\n"
    for r in df2.head(num_matches).itertuples():
        thumb_md = f"![]({r.thumb})"
        url      = f"https://youtu.be/{r.videoId}"
        md2 += (
            f"| {thumb_md} | [{r.title}]({url}) | {r.channel} | "
            f"{r.uploadDate} | {format_views(r.views)} | "
            f"{r._13:.1f}% | {r._14:.1f}% |\n"
        )
    st.markdown(md2, unsafe_allow_html=True)

    # Table 3 – Intro Text Matches
    secs = 20 if content_type.startswith("Shorts") else 60
    intro = get_intro_text(src["videoId"], secs)
    df_cand["Intro→Title %"]     = df_cand["title"].map(lambda t: fuzz.ratio(intro, t))
    df_cand["ThumbText"]         = df_cand["thumb"].map(extract_text_via_vision)
    df_cand["Intro→ThumbText %"] = df_cand["ThumbText"].map(lambda x: fuzz.ratio(intro, x))
    df_cand["Intro Combined %"]  = df_cand[["Intro→Title %","Intro→ThumbText %"]].max(axis=1)
    df_cand.sort_values("Intro Combined %", ascending=False, inplace=True)

    st.subheader("Table 3 – Intro Text Matches")
    md3 = "| Title | Channel | Uploaded | Views | Intro→Title % | Intro→ThumbText % | Combined % |\n"
    md3 += "|---|---|---|---:|---:|---:|---:|\n"
    for r in df_cand.head(num_matches).itertuples():
        url = f"https://youtu.be/{r.videoId}"
        md3 += (
            f"| [{r.title}]({url}) | {r.channel} | {r.uploadDate} | "
            f"{format_views(r.views)} | {r._17:.1f}% | {r._18:.1f}% | {r._19:.1f}% |\n"
        )
    st.markdown(md3, unsafe_allow_html=True)
