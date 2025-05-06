import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from rapidfuzz import fuzz

st.set_page_config(layout="wide")
st.title("YouTube Re-Indexation Dashboard")

# Initialize YouTube client
API_KEY = st.secrets["YOUTUBE"]["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

@st.cache_data(show_spinner=False)
def get_channel_videos(chan_id):
    videos = []
    req = youtube.search().list(part="id", channelId=chan_id, maxResults=50, order="date", type="video")
    while req:
        res = req.execute()
        for item in res["items"]:
            videos.append(item["id"]["videoId"])
        req = youtube.search().list_next(req, res)
    return videos

@st.cache_data(show_spinner=False)
def get_video_details(vid_ids):
    details = []
    for i in range(0, len(vid_ids), 50):
        chunk = vid_ids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            details.append({
                "videoId": v["id"],
                "title": v["snippet"]["title"],
                "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                "duration": v["contentDetails"]["duration"],
                "views": int(v["statistics"].get("viewCount", 0))
            })
    return pd.DataFrame(details)

def fetch_intro_text(vid_id, chars=300):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid_id)
        text = " ".join([seg["text"] for seg in transcript])
        return text[:chars]
    except:
        return ""

def find_best_match(query, exclude_chan, duration_filter=None):
    params = dict(part="snippet", q=query, type="video", maxResults=10, order="viewCount")
    if duration_filter:
        params["videoDuration"] = duration_filter  # "short" for shorts
    res = youtube.search().list(**params).execute()
    vids = [item["id"]["videoId"] for item in res["items"] if item["snippet"]["channelId"] != exclude_chan]
    if not vids: return None
    stats = get_video_details(vids)
    return stats.sort_values("views", ascending=False).iloc[0].to_dict()

# Sidebar inputs
chan_id = st.sidebar.text_input("Your Channel ID", "")
intro_thresh = st.sidebar.slider("Intro Similarity Threshold (%)", 0, 100, 50)
st.sidebar.markdown("**Steps:**\n1. Enter your Channel ID\n2. Wait for data to load")

if chan_id:
    with st.spinner("Fetching your channel videos…"):
        all_vids = get_channel_videos(chan_id)
    df = get_video_details(all_vids)
    df_long = df[~df["duration"].str.contains("M0S")]  # long‐form heuristic
    df_shorts = df[df["duration"].str.contains("M0S")]  # shorts heuristic

    tab1, tab2 = st.tabs(["Title/Thumbnail Matches", "Shorts Matches"])

    with tab1:
        st.header("Long-form Video Matches")
        rows = []
        for _, row in df_long.iterrows():
            match = find_best_match(row["title"], chan_id)
            if not match: continue
            our_intro = fetch_intro_text(row["videoId"])
            their_intro = fetch_intro_text(match["videoId"])
            sim = fuzz.ratio(our_intro, their_intro)
            rows.append({
                "Our Title": row["title"],
                "Their Title": match["title"],
                "Our Thumbnail": row["thumbnail"],
                "Their Thumbnail": match["thumbnail"],
                "Their Views": match["views"],
                "Intro Sim (%)": sim
            })
        out = pd.DataFrame(rows)
        st.dataframe(out.style.format({"Intro Sim (%)": "{:.0f}"}), use_container_width=True)

    with tab2:
        st.header("Shorts Topic Matches")
        rows2 = []
        for _, row in df_shorts.iterrows():
            match = find_best_match(row["title"], chan_id, duration_filter="short")
            if not match: continue
            rows2.append({
                "Our Short Title": row["title"],
                "Their Short Title": match["title"],
                "Their Thumbnail": match["thumbnail"],
                "Their Views": match["views"]
            })
        out2 = pd.DataFrame(rows2)
        st.dataframe(out2, use_container_width=True)
