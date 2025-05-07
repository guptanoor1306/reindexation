import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from googleapiclient.discovery import build
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Only search within these 83 channels ──
ALLOWED_CHANNELS = [
    "UCK7tptUDHh-RYDsdxO1-5QQ","UCvJJ_dzjViJCoLf5uKUTwoA","UCvQECJukTDE2i6aCoMnS-Vg",
    # … (other IDs here) …
    "UCczAxLCL79gHXKYaEc9k-ZQ","UCqykZoZjaOPb6i_Y5gk0kLQ",
]

# ── Secrets & clients ──
YT_KEY     = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
VISION_KEY = st.secrets["VISION"]["API_KEY"]
youtube    = build("youtube", "v3", developerKey=YT_KEY)
openai_cli = OpenAI(api_key=OPENAI_KEY)

# ── Helpers ──
def parse_iso_duration(dur):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    return (int(m.group(1) or 0)*3600 +
            int(m.group(2) or 0)*60 +
            int(m.group(3) or 0))

def format_views(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >=   1_000: return f"{n/1_000:.1f}K"
    return str(n)

@st.cache_data
def fetch_my_videos(cid):
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
def fetch_video_details(vids):
    rows = []
    for i in range(0, len(vids), 50):
        chunk = vids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            pub = v["snippet"]["publishedAt"]
            rows.append({
                "videoId":    v["id"],
                "title":      v["snippet"]["title"],
                "channel":    v["snippet"]["channelTitle"],
                "uploadDate": datetime.fromisoformat(pub.rstrip("Z")).date().isoformat(),
                "thumb":      v["snippet"]["thumbnails"]["high"]["url"],
                "views":      int(v["statistics"].get("viewCount", 0)),
                "type":       "Short" if sec <= 180 else "Long-Form"
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text):
    resp = openai_cli.embeddings.create(model="text-embedding-ada-002", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim(a, b):
    return float((a @ b) / (np.linalg.norm(a)*np.linalg.norm(b))) * 100.0

def extract_text_via_vision(url):
    endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_KEY}"
    body = {"requests":[{"image":{"source":{"imageUri":url}},
                         "features":[{"type":"TEXT_DETECTION","maxResults":1}]}]}
    r = requests.post(endpoint, json=body).json()
    try: return r["responses"][0]["fullTextAnnotation"]["text"]
    except: return ""

def get_intro_text(video_id, seconds):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text, total = "", 0
        for seg in transcript:
            if total >= seconds:
                break
            text += " " + seg.get("text", "")
            total += seg.get("duration", 0)
        return text.strip()
    except:
        return ""

# ── Sidebar ──
channel_id   = st.sidebar.text_input("Your Channel ID")
content_type = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤ 3 min)"])
num_matches  = st.sidebar.number_input("Results to show", 1, 10, 5)

if not channel_id:
    st.info("Enter your YouTube Channel ID."); st.stop()

# ── Load & filter uploads ──
with st.spinner("Loading your uploads…"):
    my_ids = fetch_my_videos(channel_id)
if not my_ids:
    st.error("No videos found."); st.stop()

df_all     = fetch_video_details(my_ids)
want_short = content_type.startswith("Shorts")
df         = df_all[df_all["type"] == ("Short" if want_short else "Long-Form")]
if df.empty:
    st.warning(f"No {content_type} found."); st.stop()

# ── Select source video ──
st.subheader("1) Select one of your videos")
sel = st.selectbox("Your videos", df["title"].tolist())
src = df[df["title"] == sel].iloc[0]

st.image(src["thumb"], caption=f"▶️ {sel}", width=300)
st.markdown(
    f"**Channel:** {src['channel']}  "
    f"**Uploaded:** {src['uploadDate']}  "
    f"**Views:** {format_views(src['views'])}"
)

st.subheader("2) Enter a primary keyword (mandatory)")
pk = st.text_input("Primary keyword")
if not pk:
    st.info("Enter a primary keyword."); st.stop()

# ── Precompute embeddings & visuals ──
emb_src  = get_embedding(src["title"])
text_src = extract_text_via_vision(src["thumb"])
img      = Image.open(requests.get(src["thumb"], stream=True).raw)\
               .convert("RGB").resize((256,256))
hist_src = img.histogram(); total = sum(hist_src)
def hist_sim(url):
    i = Image.open(requests.get(url, stream=True).raw)\
             .convert("RGB").resize((256,256))
    h = i.histogram()
    inter = sum(min(hist_src[j], h[j]) for j in range(len(h)))
    return inter/total*100

if st.button("3) Run Title & Thumbnail & Intro Match"):
    def yt_search(q):
        return requests.get(
            "https://youtube.googleapis.com/youtube/v3/search",
            params=dict(part="snippet",q=q,type="video",
                        order="viewCount",maxResults=50,key=YT_KEY)
        ).json().get("items",[])

    # semantic + keyword candidates
    sem = yt_search(src["title"])
    cand_sem = [i["id"]["videoId"] for i in sem if i["snippet"]["channelId"] in ALLOWED_CHANNELS]
    key = yt_search(pk)
    cand_key = [i["id"]["videoId"] for i in key if i["snippet"]["channelId"] in ALLOWED_CHANNELS]
    combined = list(dict.fromkeys(cand_sem + cand_key))
    if not combined:
        st.warning("No matches found."); st.stop()

    df_cand = fetch_video_details(combined)
    # Title-match
    df_cand["sem"]   = df_cand["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
    df_cand["key"]   = df_cand["title"].map(lambda t: fuzz.ratio(pk, t))
    df_cand["score"] = df_cand[["sem","key"]].max(axis=1)
    df_cand.sort_values("score", ascending=False, inplace=True)

    # ── Table 1: Titles ──
    st.subheader("Table 1 – Title Matches")
    top1 = df_cand.head(num_matches)
    md1 = "| Title | Channel | Uploaded | Views | Sem % | Key % | Combined % |\n"
    md1 += "| --- | --- | --- | ---: | ---: | ---: | ---: |\n"
    for r in top1.itertuples():
        url = f"https://youtu.be/{r.videoId}"
        md1 += (
            f"| [{r.title}]({url}) | {r.channel} | {r.uploadDate} | "
            f"{format_views(r.views)} | {r.sem:.1f}% | {r.key:.1f}% | {r.score:.1f}% |\n"
        )
    st.markdown(md1, unsafe_allow_html=True)

    # ── Table 2: Thumbnails ──
    st.subheader("Table 2 – Thumbnail Matches")
    df_cand["text"] = df_cand["thumb"].map(lambda u: fuzz.ratio(text_src, extract_text_via_vision(u)))
    df_cand["hist"] = df_cand["thumb"].map(hist_sim)
    df2 = df_cand[(df_cand["text"]>0)|(df_cand["hist"]>0)].copy()
    df2.sort_values(["hist","text"], ascending=[False,False], inplace=True)
    top2 = df2.head(num_matches)

    md2 = "| Thumbnail | Title | Channel | Uploaded | Views | Text % | Visual % |\n"
    md2 += "| :---: | --- | --- | :---: | ---: | ---: | ---: |\n"
    for r in top2.itertuples():
        thumb_md = f"![]({r.thumb})"
        url      = f"https://youtu.be/{r.videoId}"
        md2 += (
            f"| {thumb_md} | [{r.title}]({url}) | {r.channel} | "
            f"{r.uploadDate} | {format_views(r.views)} | "
            f"{r.text:.1f}% | {r.hist:.1f}% |\n"
        )
    st.markdown(md2, unsafe_allow_html=True)

    # ── Table 3: Intro Text ──
    st.subheader("Table 3 – Intro Text Matches")
    secs = 20 if want_short else 60
    intro = get_intro_text(src["videoId"], secs)
    df_cand["intro_title_sim"] = df_cand["title"].map(lambda t: fuzz.ratio(intro, t))
    df_cand["thumb_text"]      = df_cand["thumb"].map(lambda u: extract_text_via_vision(u))
    df_cand["intro_thumb_sim"] = df_cand["thumb_text"].map(lambda x: fuzz.ratio(intro, x))
    df_cand["intro_score"]     = df_cand[["intro_title_sim","intro_thumb_sim"]].max(axis=1)
    top3 = df_cand.sort_values("intro_score", ascending=False).head(num_matches)

    md3 = "| Title | Channel | Uploaded | Views | Intro→Title % | Intro→ThumbText % | Combined % |\n"
    md3 += "| --- | --- | --- | ---: | ---: | ---: | ---: |\n"
    for r in top3.itertuples():
        url = f"https://youtu.be/{r.videoId}"
        md3 += (
            f"| [{r.title}]({url}) | {r.channel} | {r.uploadDate} | "
            f"{format_views(r.views)} | {r.intro_title_sim:.1f}% | "
            f"{r.intro_thumb_sim:.1f}% | {r.intro_score:.1f}% |\n"
        )
    st.markdown(md3, unsafe_allow_html=True)
