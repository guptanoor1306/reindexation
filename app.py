import os
import re
import requests
import subprocess
import tempfile
import shutil
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from googleapiclient.discovery import build
from openai import OpenAI
from rapidfuzz import fuzz
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔍 Zero1 YouTube Title & Thumbnail Matcher")

# ── Hard-coded channel ID ──
CHANNEL_ID = "UCUUlw3anBIkbW9W44Y-eURw"
st.sidebar.markdown(f"**Channel ID:** `{CHANNEL_ID}`")

# ── Allowed 83 channels ──
ALLOWED_CHANNELS = [
    "UCK7tptUDHh-RYDsdxO1-5QQ","UCvJJ_dzjViJCoLf5uKUTwoA","UCvQECJukTDE2i6aCoMnS-Vg",
    "UCJFp8uSYCjXOMnkUyb3CQ3Q","UCUyDOdBWhC1MCxEjC46d-zw","UCWHCXSKASuSzao_pplQ7SPw",
    "UCw5TLrz3qADabwezTEcOmgQ","UC415bOPUcGSamy543abLmRA","UCRzYN32xtBf3Yxsx5BvJWJw",
    "UCLXo7UDZvByw2ixzpQCufnA","UCMiJRAwDNSNzuYeN2uWa0pA","UCBJycsmduvYEL83R_U4JriQ",
    "UCVOTBwF0vnSxMRIbfSE_K_g","UCSPYNpQ2fHv9HJ-q6MIMaPw","UCUMccND2H_CVS0dMZKCPCXA",
    "UCEhBVAPy-bxmnbNARF-_tvA","UCQQojT_AmVWGb4Eg-QniuBA","UCtinbF-Q-fVthA0qrFQTgXQ",
    "UCV6KDgJskWaEckne5aPA0aQ","UCoOae5nYA7VqaXzerajD0lg","UCPgfM-dk3XAb4T3DtT6Nwsw",
    "UCnpekFV93kB1O0rVqEKSumg","UC7ZddA__ewP3AtDefjl_tWg","UC3mjMoJuFnjYRBLon_6njbQ",
    "UCqW8jxh4tH1Z1sWPbkGWL4g","UC3DkFux8Iv-aYnTRWzwaiBA","UCsNxHPbaCWL1tKw2hxGQD6g",
    "UCPk2s5c4R_d-EUUNvFFODoA","UCwVEhEzsjLym_u1he4XWFkg","UCvs2mwDS-ZiIeJ01kvzarbQ",
    "UCAxUtcgLiq_gopO87VaZM5w","UCwAdQUuPT6laN-AQR17fe1g","UC80Voenx9LIHY7TNwz55x7w",
    "UCBqvATpjSubtNxpqUDj4_cA","UCvqttS8EzhRq2YWg03qKRCQ","UCODr9HUJ90xtWD-0Xoz4vPw",
    "UCe6eisvsctSPvBhmincn6kA","UCA295QVkf9O1RQ8_-s3FVXg","UC4QZ_LsYcvcq7qOsOhpAX4A",
    "UCkw1tYo7k8t-Y99bOXuZwhg","UCQXwgooTlP6tk2a-u6vgyUA","UCB7GnQlJPIL6rBBqEoX87vA",
    "UCmGSJVG3mCRXVOP4yZrU1Dw","UC0a_pO439rhcyHBZq3AKdrw","UCJ24N4O0bP7LGLBDvye7oCA",
    "UCHnyfMqiRRG1u-2MsSQLbXA","UCvK4bOhULCpmLabd2pDMtnA","UCXbKJML9pVclFHLFzpvBgWw",
    "UCnmGIkw-KdI0W5siakKPKog","UCWpk9PSGHoJW1hZT4egxTNQ","UCGq-a57w-aPwyi3pW7XLiHw",
    "UCL_v4tC26PvOFytV1_eEVSg","UCE4Gn00XZbpWvGUfIslT-tA","UCm5iBOcQ0GET_2FqFI61QDA",
    "UCLQOtbB1COQwjcCEPB2pa8w","UCqit4NtRDfdEHKX_zgmAwrg","UCkCGANrihzExmu9QiqZpPlQ",
    "UC9RM-iSvTu1uPJb8X5yp3EQ","UCdCottK2mn8T7VOHleKCYCg","UCxgAuX3XZROujMmGphN_scA",
    "UCY1kMZp36IQSyNx_9h4mpCg","UCO3tlaeZ6Z0ZN5frMZI3-uQ","UCf_XYgupvdx7rA44Ap3uI5w",
    "UCtnItzU7q_bA1eoEBjqcVrw","UCgNg3vwj3xt7QOrcIDaHdFg","UCggPd3Vf9ooG2r4I_ZNWBzA",
    "UCQpPo9BNwezg54N9hMFQp6Q","UCvcEBQ0K3UsQ8bzWKHKQmbw","UCFDxyA1H3VEN0VQwfMe2VMQ",
    "UCVRqLKnUgC4BM3Vu7gZYQcw","UC8uj-UFGDzAx3RfPzeRqnyA","UC7KbIaEOuY7H2j-cvhJ3mYA",
    "UCvBy3qcISSOcrbqPhqmG4Xw","UCAL3JXZSzSm8AlZyD3nQdBA","UCtYKe7-XbaDjpUwcU5x0bLg",
    "UCODHrzPMGbNv67e84WDZhQQ","UCkjrBN_GAjFJyVvjcI07KkQ","UCii9ezsUa_mBiSdw0PtSOaw",
    "UCR0tBVaZPaSqmdqkw7oYmcw","UCPjHhJ3fxgcV5Gv5uVAhNEA","UCT0dmfFCLWuVKPWZ6wcdKyg",
    "UCczAxLCL79gHXKYaEc9k-ZQ","UCqykZoZjaOPb6i_Y5gk0kLQ"
]

# ── API clients ──
YT_KEY     = st.secrets["YOUTUBE"]["API_KEY"]
OPENAI_KEY = st.secrets["OPENAI"]["API_KEY"]
VISION_KEY = st.secrets["VISION"]["API_KEY"]
youtube    = build("youtube", "v3", developerKey=YT_KEY)
openai_cli = OpenAI(api_key=OPENAI_KEY)

# ── Utility funcs ──
def parse_iso_duration(dur):  # ISO8601 → seconds
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    return (int(m.group(1) or 0)*3600 +
            int(m.group(2) or 0)*60 +
            int(m.group(3) or 0))

def format_views(n):
    if n>=1_000_000: return f"{n/1_000_000:.1f}M"
    if n>=  1_000: return f"{n/1_000:.1f}K"
    return str(n)

@st.cache_data
def fetch_my_videos(cid):
    ids, req = [], youtube.search().list(
        part="id", channelId=cid, type="video",
        order="date", maxResults=50
    )
    while req:
        res = req.execute()
        ids += [i["id"]["videoId"] for i in res.get("items",[])]
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
        for v in resp.get("items",[]):
            sec = parse_iso_duration(v["contentDetails"]["duration"])
            pub = v["snippet"]["publishedAt"]
            rows.append({
                "videoId":    v["id"],
                "title":      v["snippet"]["title"],
                "type":       "Short" if sec<=180 else "Long-Form",
                "channel":    v["snippet"]["channelTitle"],
                "uploadDate": datetime.fromisoformat(pub.rstrip("Z")).date().isoformat(),
                "thumb":      v["snippet"]["thumbnails"]["high"]["url"],
                "views":      int(v["statistics"].get("viewCount",0))
            })
    return pd.DataFrame(rows)

@st.cache_data
def get_embedding(text):
    resp = openai_cli.embeddings.create(model="text-embedding-ada-002", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim(a,b):
    return float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))*100.0

def extract_text_via_vision(url):
    r = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={VISION_KEY}",
        json={"requests":[{"image":{"source":{"imageUri":url}},
                           "features":[{"type":"TEXT_DETECTION","maxResults":1}]}]}
    ).json()
    try: return r["responses"][0]["fullTextAnnotation"]["text"]
    except: return ""

@st.cache_data(show_spinner=False)
def get_intro_text(vid, secs):
    tdir = tempfile.mkdtemp()
    out  = os.path.join(tdir,f"{vid}.webm")
    try:
        subprocess.run([
            "yt-dlp","-f","bestaudio",
            "--download-sections",f"*00:00:00-00:00:{secs:02d}",
            "-o",out,f"https://youtu.be/{vid}"
        ], check=True)
        with open(out,"rb") as f:
            return openai_cli.audio.transcriptions.create(model="whisper-1", file=f).text
    except: 
        return ""
    finally:
        shutil.rmtree(tdir,ignore_errors=True)

# ── Sidebar ──
content_type = st.sidebar.selectbox("Filter by:",["Long-Form (>3m)","Shorts (≤3m)"])
num_matches  = st.sidebar.number_input("Results to show",1,10,5)

# ── Load your uploads ──
with st.spinner("Loading your uploads…"):
    my_ids = fetch_my_videos(CHANNEL_ID)
df_all = fetch_video_details(my_ids)
if df_all.empty:
    st.error("No uploads found."); st.stop()

want_short = content_type.startswith("Shorts")
df = df_all[df_all["type"]==( "Short" if want_short else "Long-Form" )]
if df.empty:
    st.warning("No videos of that type."); st.stop()

# ── 1) pick your video ──
st.subheader("1) Choose one of your videos")
sel = st.selectbox("Your videos",df["title"])
src = df[df["title"]==sel].iloc[0]
st.image(src["thumb"],width=320,caption=sel)
st.caption(f"{src['channel']} · {src['uploadDate']} · {format_views(src['views'])} views")

# ── 2) primary keyword ──
st.subheader("2) Enter primary keyword")
pk = st.text_input("Keyword")
if not pk:
    st.info("Please enter a keyword."); st.stop()

# ── precompute ──
emb_src  = get_embedding(src["title"])
text_src = extract_text_via_vision(src["thumb"])
img      = Image.open(requests.get(src["thumb"],stream=True).raw)\
            .convert("RGB").resize((256,256))
hist_src = img.histogram(); total=sum(hist_src)
def hist_sim(u):
    i2       = Image.open(requests.get(u,stream=True).raw)\
                 .convert("RGB").resize((256,256))
    h2       = i2.histogram()
    intersect= sum(min(hist_src[i],h2[i]) for i in range(len(h2)))
    return intersect/total*100.0

debug_logs=[]
if st.button("3) Run Matches"):

    def yt_search(q,dur=None):
        p=dict(part="snippet",q=q,type="video",order="viewCount",
               maxResults=50,key=YT_KEY)
        if dur: p["videoDuration"]=dur
        debug_logs.append(f"search {q!r} dur={dur}: {p}")
        items=requests.get("https://youtube.googleapis.com/youtube/v3/search",params=p).json().get("items",[])
        debug_logs.append(f"→ {len(items)} items")
        return items

    # gather IDs
    sem_all    = yt_search(src["title"])
    sem_sh     = yt_search(src["title"],"short")
    key_all    = yt_search(pk)
    key_sh     = yt_search(pk,"short")
    cand_sem   = [i["id"]["videoId"] 
                  for i in (sem_all+sem_sh) if i["snippet"]["channelId"] in ALLOWED_CHANNELS]
    cand_key   = [i["id"]["videoId"] 
                  for i in (key_all+key_sh) if i["snippet"]["channelId"] in ALLOWED_CHANNELS]
    combined   = list(dict.fromkeys(cand_sem+ cand_key))
    debug_logs.append(f"combined ids ({len(combined)}):{combined}")
    if not combined:
        st.warning("No matches found.")
    df_cand = fetch_video_details(combined)

    # metrics
    df_cand["sem_pct"]      = df_cand["title"].map(lambda t:cosine_sim(emb_src,get_embedding(t)))
    df_cand["key_pct"]      = df_cand["title"].map(lambda t:fuzz.ratio(pk,t))
    df_cand["combined_pct"] = df_cand[["sem_pct","key_pct"]].max(axis=1)
    df_cand["text_pct"]     = df_cand["thumb"].map(lambda u:fuzz.ratio(text_src,extract_text_via_vision(u)))
    df_cand["visual_pct"]   = df_cand["thumb"].map(hist_sim)
    secs=20 if want_short else 60
    intro=get_intro_text(src["videoId"],secs)
    debug_logs.append(f"intro len {len(intro)}")
    if intro:
        df_cand["intro_title_pct"]     = df_cand["title"].map(lambda t:fuzz.ratio(intro,t))
        df_cand["intro_thumb_pct"]     = df_cand["thumb"].map(lambda u:fuzz.ratio(intro,extract_text_via_vision(u)))
        df_cand["intro_combined_pct"]  = df_cand[["intro_title_pct","intro_thumb_pct"]].max(axis=1)
    else:
        df_cand[["intro_title_pct","intro_thumb_pct","intro_combined_pct"]]=0

    # helper to render cards
    def render_cards(rows, metrics):
        per_row=4
        for idx,r in enumerate(rows):
            if idx%per_row==0:
                cols = st.columns(per_row)
            c = cols[idx%per_row]
            with c:
                st.image(r["thumb"],use_column_width=True)
                st.markdown(f"**[{r['title']}](https://youtu.be/{r['videoId']})**")
                st.caption(f"{r['channel']} · {r['uploadDate']} · {format_views(r['views'])} views")
                for k in metrics:
                    st.markdown(f"- **{k}**: {r[k]:.1f}%")

    # Table 1 cards
    st.subheader("Table 1 – Title Matches")
    t1l,t1s = st.tabs(["Long-Form","Shorts"])
    for tab,v in ((t1l,"Long-Form"),(t1s,"Short")):
        with tab:
            sub = df_cand[df_cand["type"]==v]
            top = sub.nlargest(num_matches,"combined_pct")
            render_cards(top.to_dict("records"), ["sem_pct","key_pct","combined_pct"])

    # Table 2 cards
    st.subheader("Table 2 – Thumbnail Matches")
    t2l,t2s = st.tabs(["Long-Form","Shorts"])
    for tab,v in ((t2l,"Long-Form"),(t2s,"Short")):
        with tab:
            sub = df_cand[(df_cand["type"]==v)&((df_cand["text_pct"]>0)|(df_cand["visual_pct"]>0))]
            top = sub.nlargest(num_matches,["visual_pct","text_pct"])
            render_cards(top.to_dict("records"), ["text_pct","visual_pct"])

    # Table 3 cards
    st.subheader("Table 3 – Intro Text Matches")
    if not intro:
        st.warning("No intro transcript.")
    else:
        t3l,t3s = st.tabs(["Long-Form","Shorts"])
        for tab,v in ((t3l,"Long-Form"),(t3s,"Short")):
            with tab:
                sub = df_cand[df_cand["type"]==v]
                top = sub.nlargest(num_matches,"intro_combined_pct")
                render_cards(top.to_dict("records"), ["intro_title_pct","intro_thumb_pct","intro_combined_pct"])

    # debug logs
    with st.expander("Debug logs"):
        for ln in debug_logs:
            st.text(ln)
