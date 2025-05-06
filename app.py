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

# — Initialize YouTube API client —
API_KEY = st.secrets["YOUTUBE"]["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

# — Helpers —
def parse_duration_to_seconds(dur):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    h, mm, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h*3600 + mm*60 + s

@st.cache_data(show_spinner=False)
def get_channel_video_ids(channel_id):
    ids = []
    req = youtube.search().list(part="id", channelId=channel_id,
                                maxResults=50, order="date", type="video")
    while req:
        res = req.execute()
        ids += [item["id"]["videoId"] for item in res["items"]]
        req = youtube.search().list_next(req, res)
    return ids

@st.cache_data(show_spinner=False)
def get_video_snippets(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = youtube.videos().list(part="snippet,contentDetails",
                                     id=",".join(chunk)).execute()
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

@st.cache_data(show_spinner=False)
def get_video_stats(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = youtube.videos().list(part="snippet,statistics",
                                     id=",".join(chunk)).execute()
        for v in resp["items"]:
            rows.append({
                "videoId": v["id"],
                "title": v["snippet"]["title"],
                "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                "views": int(v["statistics"].get("viewCount", 0))
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def fetch_intro_text(video_id, chars=300):
    try:
        segs = YouTubeTranscriptApi.get_transcript(video_id)
        txt = " ".join(s["text"] for s in segs)
        return txt[:chars]
    except:
        return ""

@st.cache_data(show_spinner=False)
def get_image_hash(url):
    resp = requests.get(url, stream=True)
    img = Image.open(resp.raw).convert("RGB")
    return imagehash.phash(img)

@st.cache_data(show_spinner=False)
def find_semantic_matches(video_id, exclude_channel, duration_filter, top_n):
    params = {
        "part": "snippet",
        "relatedToVideoId": video_id,
        "type": "video",
        "maxResults": top_n
    }
    if duration_filter:
        params["videoDuration"] = duration_filter  # "short" or omit
    res = youtube.search().list(**params).execute()
    vids = [
        it["id"]["videoId"]
        for it in res["items"]
        if it["snippet"]["channelId"] != exclude_channel
    ]
    if not vids:
        return []
    stats = get_video_stats(vids)
    return stats.sort_values("views", ascending=False).to_dict("records")

# — Sidebar Controls —
channel_id = st.sidebar.text_input("Your Channel ID")
num_matches = st.sidebar.number_input("Matches to fetch", 1, 10, 5)
title_thresh = st.sidebar.slider("Title similarity ≥ (%)", 0, 100, 50)
intro_thresh = st.sidebar.slider("Intro similarity ≥ (%)", 0, 100, 50)
thumb_thresh = st.sidebar.slider("Max thumbnail hash dist", 0, 64, 10)
video_type = st.sidebar.selectbox("Video type", ["Long-form", "Shorts"])

# — Main Logic —
if channel_id:
    with st.spinner("Loading your videos…"):
        ids = get_channel_video_ids(channel_id)
        df = get_video_snippets(ids)

    subset = df[df["is_short"] == (video_type == "Shorts")]
    choice = st.selectbox(f"Select a {video_type} video", subset["title"].tolist())

    if st.button("Run Similarity Search"):
        src = subset[subset["title"] == choice].iloc[0]
        our_intro = fetch_intro_text(src["videoId"])
        our_hash  = get_image_hash(src["thumbnail"])

        results = find_semantic_matches(
            src["videoId"],
            channel_id,
            duration_filter="short" if video_type=="Shorts" else None,
            top_n=num_matches
        )

        if not results:
            st.warning("No semantic matches found.")
        else:
            # build markdown table
            md = (
                "| Their Video | Thumbnail | Views | Title | Thumb | Intro |\n"
                "|---|---|---|:---:|:---:|:---:|\n"
            )
            for m in results:
                t_sim = fuzz.ratio(choice, m["title"])
                t_ok  = "✅" if t_sim >= title_thresh else "❌"

                their_hash = get_image_hash(m["thumbnail"])
                h_dist = our_hash - their_hash
                h_ok   = "✅" if h_dist <= thumb_thresh else "❌"

                their_intro = fetch_intro_text(m["videoId"])
                i_sim = fuzz.ratio(our_intro, their_intro)
                i_ok  = "✅" if i_sim >= intro_thresh else "❌"

                link = f"https://youtube.com/watch?v={m['videoId']}"
                md += (
                    f"| [{m['title']}]({link}) "
                    f"| [![]({m['thumbnail']})]({link}) "
                    f"| {m['views']:,} "
                    f"| {t_ok} "
                    f"| {h_ok} "
                    f"| {i_ok} |\n"
                )
            st.markdown(md)
