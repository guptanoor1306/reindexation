# app.py
import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from rapidfuzz import fuzz

st.set_page_config(layout="wide")
st.title("YouTube Re-Indexation Dashboard")

# YouTube client
API_KEY = st.secrets["YOUTUBE"]["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

@st.cache_data(show_spinner=False)
def get_channel_video_ids(channel_id):
    ids = []
    req = youtube.search().list(
        part="id",
        channelId=channel_id,
        maxResults=50,
        order="date",
        type="video"
    )
    while req:
        res = req.execute()
        ids += [item["id"]["videoId"] for item in res["items"]]
        req = youtube.search().list_next(req, res)
    return ids

@st.cache_data(show_spinner=False)
def get_video_snippet_details(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            rows.append({
                "videoId": v["id"],
                "title": v["snippet"]["title"],
                "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                "duration": v["contentDetails"]["duration"],
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_video_stats(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            rows.append({
                "videoId": v["id"],
                "title": v["snippet"]["title"],
                "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                "views": int(v["statistics"].get("viewCount", 0)),
            })
    return pd.DataFrame(rows)

def fetch_intro_text(video_id, chars=300):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(seg["text"] for seg in transcript)
        return text[:chars]
    except:
        return ""

def find_best_match(query, exclude_channel, duration_filter=None):
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 10,
        "order": "viewCount",
    }
    if duration_filter:
        params["videoDuration"] = duration_filter  # "short" for shorts
    res = youtube.search().list(**params).execute()
    candidates = [
        item["id"]["videoId"]
        for item in res["items"]
        if item["snippet"]["channelId"] != exclude_channel
    ]
    if not candidates:
        return None
    stats = get_video_stats(candidates)
    return stats.sort_values("views", ascending=False).iloc[0].to_dict()

# --- Sidebar: Channel ID input ---
channel_id = st.sidebar.text_input("Enter your Channel ID")
if channel_id:
    with st.spinner("Loading video metadata…"):
        vid_ids = get_channel_video_ids(channel_id)
        df = get_video_snippet_details(vid_ids)

    # classify long-form vs shorts
    df["is_short"] = df["duration"].str.contains("M0S")

    video_type = st.selectbox(
        "Choose video type",
        ["Long-form", "Shorts"]
    )
    subset = df[df["is_short"] == (video_type == "Shorts")]
    titles = subset["title"].tolist()

    selected_title = st.selectbox(f"Select a {video_type} video", titles)

    if st.button("Run Similarity Search"):
        vid_row = subset[subset["title"] == selected_title].iloc[0]
        match = find_best_match(
            selected_title,
            channel_id,
            duration_filter="short" if video_type == "Shorts" else None
        )

        if not match:
            st.warning("No external match found.")
        else:
            st.subheader("Best External Match")
            c1, c2 = st.columns(2)
            c1.image(vid_row["thumbnail"], use_column_width=True, caption="Your Video")
            c2.image(match["thumbnail"], use_column_width=True, caption="Matched Video")

            st.markdown(f"**Their Title:** {match['title']}")
            st.markdown(f"**Their Views:** {match['views']:,}")

            if video_type == "Long-form":
                our_intro = fetch_intro_text(vid_row["videoId"])
                their_intro = fetch_intro_text(match["videoId"])
                sim = fuzz.ratio(our_intro, their_intro)
                st.markdown(f"**Intro Similarity:** {sim}%")
