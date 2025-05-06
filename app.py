import json
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.cloud import vision
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image
import imagehash

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Load secrets and initialize clients ──
YT_KEY       = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY   = st.secrets["OPENAI"]["API_KEY"]
gcp_sa_json  = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
gcp_creds    = service_account.Credentials.from_service_account_info(gcp_sa_json)
vision_cli   = vision.ImageAnnotatorClient(credentials=gcp_creds)
youtube      = build("youtube", "v3", developerKey=YT_KEY)
openai_cli   = OpenAI(api_key=OPENAI_KEY)

# ── Helper functions ──
def parse_iso_duration(dur: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    h, mm, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h * 3600 + mm * 60 + s

@st.cache_data
def fetch_my_videos(channel_id: str) -> list[str]:
    ids = []
    req = youtube.search().list(
        part="id", channelId=channel_id,
        type="video", order="date", maxResults=50
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
        chunk = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId":     v["id"],
                "title":       v["snippet"]["title"],
                "description": v["snippet"].get("description", "") or "",
                "thumb":       v["snippet"]["thumbnails"]["high"]["url"],
                "views":       int(v["statistics"].get("viewCount", 0)),
                "type":        "Short" if sec <= 180 else "Long-Form"
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    resp = openai_cli.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    emb = resp.data[0].embedding
    return np.array(emb, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 100.0

@st.cache_data
def hash_image(url: str) -> imagehash.ImageHash:
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return imagehash.phash(img)

def extract_text_cloud_vision(url: str) -> str:
    image = vision.Image()
    image.source.image_uri = url
    resp = vision_cli.text_detection(image=image)
    texts = resp.text_annotations
    return texts[0].description if texts else ""

# ── Sidebar inputs ──
channel_id      = st.sidebar.text_input("Your Channel ID")
content_type    = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤3 min)"])
num_matches     = st.sidebar.number_input("Results to show", 1, 10, 5)
primary_keyword = st.sidebar.text_input("Primary keyword (fallback)")

if channel_id:
    # 1) Load your uploads
    with st.spinner("Loading your uploads…"):
        vid_ids = fetch_my_videos(channel_id)
        df      = fetch_video_details(vid_ids)

    # 2) Filter by duration bucket
    want_short = content_type.startswith("Shorts")
    df         = df[df["type"] == ("Short" if want_short else "Long-Form")]

    # 3) Select one of your videos
    sel_title = st.selectbox("Pick one of your videos", df["title"])
    src       = df[df["title"] == sel_title].iloc[0]

    # Precompute source embedding & thumbnail hash & OCR text
    emb_src   = get_embedding(src["title"])
    hash_src  = hash_image(src["thumb"])
    text_src  = extract_text_cloud_vision(src["thumb"])

    if st.button("Run Title & Thumbnail Match"):
        # ── Table 1: Title Matching ──
        # a) Semantic search on your title
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part":       "snippet",
                "q":          src["title"],
                "type":       "video",
                "order":      "viewCount",
                "maxResults": 50,
                "key":        YT_KEY
            }
        ).json().get("items", [])
        cand_ids = [
            it["id"]["videoId"]
            for it in resp
            if it["snippet"]["channelId"] != channel_id
        ]
        details = fetch_video_details(cand_ids)
        details["sem_sim"] = details["title"].map(
            lambda t: cosine_sim(emb_src, get_embedding(t))
        )
        details = details.sort_values("sem_sim", ascending=False)

        # b) Fallback keyword search if all semantic sims are zero
        if details["sem_sim"].max() == 0 and primary_keyword:
            resp     = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part":       "snippet",
                    "q":          primary_keyword,
                    "type":       "video",
                    "order":      "viewCount",
                    "maxResults": 50,
                    "key":        YT_KEY
                }
            ).json().get("items", [])
            cand_ids = [
                it["id"]["videoId"]
                for it in resp
                if it["snippet"]["channelId"] != channel_id
            ]
            details = fetch_video_details(cand_ids)
            details["sem_sim"] = details["title"].map(
                lambda t: cosine_sim(emb_src, get_embedding(t))
            )
            details = details.sort_values("sem_sim", ascending=False)

        top1 = details.head(num_matches)

        st.subheader("Table 1 – Title Matches")
        md1 = "| Title | Type | Views | SemMatch (%) |\n|---|:---:|---:|---:|\n"
        for r in top1.itertuples():
            link = f"https://youtu.be/{r.videoId}"
            md1 += f"| [{r.title}]({link}) | {r.type} | {r.views:,} | {r.sem_sim:.1f}% |\n"
        st.markdown(md1, unsafe_allow_html=True)

        # ── Table 2: Thumbnail Matching ──
        st.subheader("Table 2 – Thumbnail Matches")
        md2 = "| Title | TextMatch (%) | VisualMatch (%) |\n|---|---:|---:|\n"
        for r in top1.itertuples():
            # OCR text match
            their_text = extract_text_cloud_vision(r.thumb)
            txt_sim    = fuzz.ratio(text_src, their_text)
            # visual hash match
            hash2      = hash_image(r.thumb)
            dist       = hash_src - hash2
            vis_sim    = max(0, (1 - dist/64) * 100)
            link       = f"https://youtu.be/{r.videoId}"
            md2       += (
                f"| [{r.title}]({link}) "
                f"| {txt_sim:.1f}% "
                f"| {vis_sim:.1f}% |\n"
            )
        st.markdown(md2, unsafe_allow_html=True)
