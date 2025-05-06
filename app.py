# app.py
import re
import requests
import streamlit as st
import pandas as pd
from PIL import Image
import imagehash
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from rapidfuzz import fuzz

st.set_page_config(layout="wide")
st.title("YouTube Re-Indexation Dashboard")

# — YouTube API setup —
API_KEY = st.secrets["YOUTUBE"]["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

def parse_duration_to_seconds(dur):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    h, mm, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h*3600 + mm*60 + s

@st.cache_data
def get_channel_video_ids(ch_id):
    ids = []
    req = youtube.search().list(
        part="id", channelId=ch_id,
        maxResults=50, order="date", type="video"
    )
    while req:
        res = req.execute()
        ids += [i["id"]["videoId"] for i in res["items"]]
        req = youtube.search().list_next(req, res)
    return ids

@st.cache_data
def get_video_snippets(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        resp = youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(video_ids[i:i+50])
        ).execute()
        for v in resp["items"]:
            sec = parse_duration_to_seconds(v["contentDetails"]["duration"])
            rows.append({
                "videoId": v["id"],
                "title": v["snippet"]["title"],
                "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                "duration_sec": sec,
                "is_short": sec <= 180
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_video_stats(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        resp = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids[i:i+50])
        ).execute()
        for v in resp["items"]:
            rows.append({
                "videoId": v["id"],
                "title": v["snippet"]["title"],
                "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                "views": int(v["statistics"].get("viewCount", 0))
            })
    return pd.DataFrame(rows)

@st.cache_data
def fetch_intro_text(video_id, chars=300):
    try:
        segs = YouTubeTranscriptApi.get_transcript(video_id)
        txt = " ".join(s["text"] for s in segs)
        return txt[:chars]
    except:
        return ""

@st.cache_data
def get_image_hash(url):
    resp = requests.get(url, stream=True)
    img = Image.open(resp.raw).convert("RGB")
    return imagehash.phash(img)

def find_semantic_matches(src_vid, ch_id, is_short, top_n):
    # 1) relatedToVideoId via REST
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "relatedToVideoId": src_vid,
        "type": "video",
        "order": "viewCount",
        "maxResults": top_n,
        "key": API_KEY
    }
    resp = requests.get(url, params=params).json()
    items = resp.get("items", [])
    vids = [it["id"]["videoId"] for it in items
            if it["snippet"]["channelId"] != ch_id]

    # 2) filter by short vs long‐form
    if not vids:
        return []
    snippets = get_video_snippets(vids)
    filtered = snippets[snippets["is_short"] == is_short]["videoId"].tolist()
    if not filtered:
        return []

    # 3) fetch stats & sort
    stats = get_video_stats(filtered)
    return stats.sort_values("views", ascending=False).to_dict("records")

def find_keyword_matches(query, ch_id, is_short, top_n):
    # fallback to q= query
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "order": "viewCount",
        "maxResults": top_n,
        "key": API_KEY
    }
    if is_short:
        params["videoDuration"] = "short"
    data = requests.get("https://www.googleapis.com/youtube/v3/search", params=params).json()
    vids = [it["id"]["videoId"] for it in data.get("items", [])
            if it["snippet"]["channelId"] != ch_id]
    if not vids:
        return []
    stats = get_video_stats(vids)
    return stats.sort_values("views", ascending=False).to_dict("records")

# — Sidebar —
channel_id   = st.sidebar.text_input("Your Channel ID")
num_matches  = st.sidebar.number_input("Matches to fetch", 1, 10, 5)
title_thresh = st.sidebar.slider("Title similarity ≥ (%)", 0, 100, 50)
intro_thresh = st.sidebar.slider("Intro similarity ≥ (%)", 0, 100, 50)
thumb_thresh = st.sidebar.slider("Max thumbnail hash dist", 0, 64, 10)
video_type   = st.sidebar.selectbox("Video type", ["Long-form", "Shorts"])

if channel_id:
    with st.spinner("Loading your videos…"):
        ids = get_channel_video_ids(channel_id)
        df  = get_video_snippets(ids)

    is_short = (video_type == "Shorts")
    subset = df[df["is_short"] == is_short]
    choice = st.selectbox(f"Select a {video_type} video", subset["title"].tolist())

    if st.button("Run Similarity Search"):
        src       = subset[subset["title"] == choice].iloc[0]
        our_intro = fetch_intro_text(src["videoId"])
        our_hash  = get_image_hash(src["thumbnail"])

        # try semantic first
        results = find_semantic_matches(src["videoId"], channel_id, is_short, num_matches)
        if not results:
            st.info("No semantic matches found; falling back to keyword search…")
            results = find_keyword_matches(choice, channel_id, is_short, num_matches)

        if not results:
            st.warning("No matches found by either method.")
        else:
            md = "| Their Video | Thumbnail | Views | Title | Thumb | Intro |\n"
            md += "|---|---|---|:---:|:---:|:---:|\n"
            for m in results:
                t_ok = "✅" if fuzz.ratio(choice, m["title"]) >= title_thresh else "❌"
                h_ok = "✅" if (our_hash - get_image_hash(m["thumbnail"])) <= thumb_thresh else "❌"
                i_ok = "✅" if fuzz.ratio(our_intro, fetch_intro_text(m["videoId"])) >= intro_thresh else "❌"
                link = f"https://youtube.com/watch?v={m['videoId']}"
                md += (
                    f"| [{m['title']}]({link}) "
                    f"| [![]({m['thumbnail']})]({link}) "
                    f"| {m['views']:,} | {t_ok} | {h_ok} | {i_ok} |\n"
                )
            st.markdown(md, unsafe_allow_html=True)
