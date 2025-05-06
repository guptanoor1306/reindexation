import re, os, tempfile, requests, streamlit as st, pandas as pd
from pytube import YouTube
from pydub import AudioSegment
from PIL import Image
import imagehash
import openai
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from rapidfuzz import fuzz

st.set_page_config(layout="wide")
st.title("YouTube Re-Indexation via Audio→Transcript Search")

# ── Initialize APIs ──
YT_KEY = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
youtube = build("youtube", "v3", developerKey=YT_KEY)
openai.api_key = OPENAI_KEY

# ── Helpers ──
def parse_iso_duration(dur):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    h, mm, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return h*3600 + mm*60 + s

@st.cache_data
def list_my_videos(ch_id):
    ids=[]
    req=youtube.search().list(part="id",channelId=ch_id,
                              type="video",order="date",maxResults=50)
    while req:
        res=req.execute()
        ids += [i["id"]["videoId"] for i in res["items"]]
        req=youtube.search().list_next(req,res)
    return ids

@st.cache_data
def fetch_snippets(ids):
    rows=[]
    for i in range(0,len(ids),50):
        resp=youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(ids[i:i+50])
        ).execute()
        for v in resp["items"]:
            sec=parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId":v["id"],
                "title":v["snippet"]["title"],
                "thumb":v["snippet"]["thumbnails"]["high"]["url"],
                "duration_sec":sec,
                "type": "Short" if sec<=180 else "Long-Form"
            })
    return pd.DataFrame(rows)

def download_first_60s_audio(vid_id):
    yt=YouTube(f"https://youtube.com/watch?v={vid_id}")
    stream=yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    tmp=stream.download(output_path=tempfile.gettempdir(),filename=f"{vid_id}.mp4")
    audio=AudioSegment.from_file(tmp).set_channels(1)
    clip=audio[:60*1000]
    wav_path=os.path.join(tempfile.gettempdir(),f"{vid_id}.wav")
    clip.export(wav_path,format="wav")
    return wav_path

def transcribe_whisper(wav_path):
    with open(wav_path,"rb") as f:
        resp=openai.Audio.transcribe("whisper-1",f)
    return resp["text"]

@st.cache_data
def search_by_query(q, top_n):
    url="https://www.googleapis.com/youtube/v3/search"
    params={
        "part":"snippet",
        "q":q,
        "type":"video",
        "order":"viewCount",
        "maxResults":top_n,
        "key":YT_KEY
    }
    data=requests.get(url,params=params).json()
    vids=[it["id"]["videoId"] for it in data.get("items",[])
          if it["snippet"]["channelId"]!=channel_id]
    if not vids: return []
    # fetch stats & details
    rows=[]
    for i in range(0,len(vids),50):
        resp=youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(vids[i:i+50])
        ).execute()
        for v in resp["items"]:
            sec=parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId":v["id"],
                "title":v["snippet"]["title"],
                "thumb":v["snippet"]["thumbnails"]["high"]["url"],
                "views":int(v["statistics"].get("viewCount",0)),
                "type": "Short" if sec<=180 else "Long-Form"
            })
    return pd.DataFrame(rows).sort_values("views",ascending=False).to_dict("records")

@st.cache_data
def hash_image(url):
    resp=requests.get(url,stream=True)
    img=Image.open(resp.raw).convert("RGB")
    return imagehash.phash(img)


# ── Sidebar ──
channel_id = st.sidebar.text_input("Your Channel ID")
num_matches = st.sidebar.number_input("How many results?",1,10,5)
title_th    = st.sidebar.slider("Title ≥ (%)",0,100,50)
thumb_th    = st.sidebar.slider("Thumb hash ≤ dist",0,64,10)
trans_th    = st.sidebar.slider("Transcript→Title ≥ (%)",0,100,50)

if channel_id:
    st.info("Fetching your video list…")
    video_ids = list_my_videos(channel_id)
    df_my = fetch_snippets(video_ids)

    sel = st.selectbox("Pick one of your videos", df_my["title"])
    src = df_my[df_my["title"]==sel].iloc[0]

    if st.button("📡 Run Audio-Transcript → Search"):
        st.spinner("Downloading & transcribing…")
        wav = download_first_60s_audio(src["videoId"])
        transcript = transcribe_whisper(wav)
        st.markdown("**Transcript (first 60s)**")
        st.write(transcript)

        st.info("Searching YouTube for that transcript…")
        candidates = search_by_query(transcript, num_matches)
        if not candidates:
            st.warning("No matches found.")
        else:
            # compute metrics
            src_hash = hash_image(src["thumb"])
            md = "| Title | Type | Views | Title✅ | Thumb✅ | Trans✅ |\n"
            md+= "|---|:---:|---:|:---:|:---:|:---:|\n"
            for c in candidates:
                t_sim = fuzz.ratio(src["title"], c["title"])
                t_ok  = "✅" if t_sim>=title_th else "❌"
                h_ok  = "✅" if (src_hash - hash_image(c["thumb"]))<=thumb_th else "❌"
                x_sim = fuzz.partial_ratio(transcript, c["title"])
                x_ok  = "✅" if x_sim>=trans_th else "❌"
                link = f"https://youtube.com/watch?v={c['videoId']}"
                md+=(
                    f"| [{c['title']}]({link}) "
                    f"| {c['type']} "
                    f"| {c['views']:,} "
                    f"| {t_ok} "
                    f"| {h_ok} "
                    f"| {x_ok} |\n"
                )
            st.markdown(md, unsafe_allow_html=True)
