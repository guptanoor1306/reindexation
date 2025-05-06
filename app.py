import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image
import imagehash

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Load secrets & init clients ──
YT_KEY     = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
VISION_KEY = st.secrets["VISION"]["API_KEY"]

youtube    = build("youtube", "v3", developerKey=YT_KEY)
openai_cli = OpenAI(api_key=OPENAI_KEY)

# ── Helpers ──
def parse_iso_duration(dur: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    return (int(m.group(1) or 0)*3600 +
            int(m.group(2) or 0)*60 +
            int(m.group(3) or 0))

@st.cache_data
def fetch_my_videos(ch_id: str) -> list[str]:
    ids=[]
    req = youtube.search().list(part="id",channelId=ch_id,
                                type="video",order="date",maxResults=50)
    while req:
        res = req.execute()
        ids += [it["id"]["videoId"] for it in res["items"]]
        req = youtube.search().list_next(req,res)
    return ids

@st.cache_data
def fetch_video_details(ids: list[str]) -> pd.DataFrame:
    rows=[]
    for i in range(0,len(ids),50):
        chunk = ids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(chunk)
        ).execute()
        for v in resp["items"]:
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            rows.append({
                "videoId": v["id"],
                "title":   v["snippet"]["title"],
                "thumb":   v["snippet"]["thumbnails"]["high"]["url"],
                "views":   int(v["statistics"].get("viewCount",0)),
                "type":    "Short" if sec<=180 else "Long-Form"
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    resp = openai_cli.embeddings.create(model="text-embedding-ada-002",input=text)
    emb  = resp.data[0].embedding
    return np.array(emb,dtype=np.float32)

def cosine_sim(a: np.ndarray,b: np.ndarray) -> float:
    return float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))*100.0

@st.cache_data
def hash_image(url: str) -> imagehash.ImageHash:
    img = Image.open(requests.get(url,stream=True).raw).convert("RGB")
    return imagehash.phash(img)

def extract_text_via_vision(url: str) -> str:
    endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_KEY}"
    body = {"requests":[
        {"image":{"source":{"imageUri":url}},
         "features":[{"type":"TEXT_DETECTION","maxResults":1}]}
    ]}
    r = requests.post(endpoint,json=body).json()
    try:
        return r["responses"][0]["fullTextAnnotation"]["text"]
    except:
        return ""

# ── Sidebar ──
channel_id      = st.sidebar.text_input("Your Channel ID")
content_type    = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)","Shorts (≤3 min)"])
num_matches     = st.sidebar.number_input("Results to show",1,10,5)
primary_keyword = st.sidebar.text_input("Primary keyword (fallback)")

if not channel_id:
    st.info("Enter your YouTube Channel ID to begin.")
    st.stop()

# 1) load and filter your uploads
with st.spinner("Loading uploads…"):
    my_ids = fetch_my_videos(channel_id)
if not my_ids:
    st.error("No videos found.")
    st.stop()

df_all = fetch_video_details(my_ids)
want_short = content_type.startswith("Shorts")
df       = df_all[df_all["type"]==("Short" if want_short else "Long-Form")]
if df.empty:
    st.warning(f"No {content_type} content found.")
    st.stop()

# 2) select source video
sel = st.selectbox("Select one of your videos",df["title"])
src = df[df["title"]==sel].iloc[0]

# precompute source signals
emb_src  = get_embedding(src["title"])
hash_src = hash_image(src["thumb"])
text_src = extract_text_via_vision(src["thumb"])

if st.button("Run Title & Thumbnail Match"):
    # ── build candidate set via semantic title search ──
    sr = requests.get("https://www.googleapis.com/youtube/v3/search",params={
        "part":"snippet","q":src["title"],"type":"video",
        "order":"viewCount","maxResults":50,"key":YT_KEY
    }).json().get("items",[])
    cand = [it["id"]["videoId"] for it in sr if it["snippet"]["channelId"]!=channel_id]
    details = fetch_video_details(cand)
    # semantic title match
    details["title_sim"] = details["title"].map(lambda t: cosine_sim(emb_src,get_embedding(t)))
    details = details.sort_values("title_sim",ascending=False)

    # fallback if all zero
    if details["title_sim"].max()==0 and primary_keyword:
        sr = requests.get("https://www.googleapis.com/youtube/v3/search",params={
            "part":"snippet","q":primary_keyword,"type":"video",
            "order":"viewCount","maxResults":50,"key":YT_KEY
        }).json().get("items",[])
        cand = [it["id"]["videoId"] for it in sr if it["snippet"]["channelId"]!=channel_id]
        details = fetch_video_details(cand)
        details["title_sim"] = details["title"].map(lambda t: cosine_sim(emb_src,get_embedding(t)))
        details = details.sort_values("title_sim",ascending=False)

    top_title = details.head(num_matches)

    # ── Table 1: Title Matches ──
    st.subheader("Table 1 – Title Matches")
    md = "| Title | Type | Views | SemMatch (%) |\n|---|:---:|---:|---:|\n"
    for r in top_title.itertuples():
        link=f"https://youtu.be/{r.videoId}"
        md+=f"| [{r.title}]({link}) | {r.type} | {r.views:,} | {r.title_sim:.1f}% |\n"
    st.markdown(md,unsafe_allow_html=True)

    # ── Table 2: Thumbnail Matches ──
    # compute text + visual match on the same candidate set
    thumb_df = details.copy()
    thumb_df["text_sim"]   = thumb_df["thumb"].map(lambda u: fuzz.ratio(text_src,extract_text_via_vision(u)))
    thumb_df["visual_sim"] = thumb_df["thumb"].map(lambda u: max(0,(1-(hash_src-hash_image(u))/64)*100))
    thumb_df = thumb_df.sort_values("visual_sim",ascending=False).head(num_matches)

    st.subheader("Table 2 – Thumbnail Matches")
    md2 = "| Thumbnail | Title | TextMatch (%) | VisualMatch (%) |\n"
    md2+= "|---|---|---:|---:|\n"
    for r in thumb_df.itertuples():
        link = f"https://youtu.be/{r.videoId}"
        # embed thumbnail image
        img_md = f"![]({r.thumb})"
        md2+=f"| {img_md} | [{r.title}]({link}) | {r.text_sim:.1f}% | {r.visual_sim:.1f}% |\n"
    st.markdown(md2,unsafe_allow_html=True)
