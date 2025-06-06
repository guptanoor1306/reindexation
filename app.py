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

# ── Only search within these 83 channels ──
ALLOWED_CHANNELS = [
    "UCK7tptUDHh-RYDsdxO1-5QQ","UCvJJ_dzjViJCoLf5uKUTwoA","UCvQECJukTDE2i6aCoMnS-Vg",
    "UCJFp8uSYCjXOMnkUyb3CQ3Q","UCUyDOdBWhC1MCxEjC46d-zw","UCWHCXSKASuSzao_pplQ7SPw",
    "UCw5TLrz3qADabwezTEcOmgQ","UC415bOPUcGSamy543abLmRA","UCRzYN32xtBf3Yxs5BvJWJw",
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

# ── Secrets & API clients ──
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

def format_views(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >=   1_000: return f"{n/1_000:.1f}K"
    return str(n)

@st.cache_data
def fetch_my_videos(cid: str) -> list[str]:
    ids, req = [], youtube.search().list(
        part="id", channelId=cid, type="video",
        order="date", maxResults=50
    )
    while req:
        res = req.execute()
        ids += [i["id"]["videoId"] for i in res.get("items", [])]
        req = youtube.search().list_next(req, res)
    return ids

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
                "type":       "Short" if sec <= 180 else "Long-Form",
                "channel":    v["snippet"]["channelTitle"],
                "uploadDate": datetime.fromisoformat(pub.rstrip("Z")).date().isoformat(),
                "thumb":      v["snippet"]["thumbnails"]["high"]["url"],
                "views":      int(v["statistics"].get("viewCount", 0))
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

@st.cache_data(show_spinner=False)
def get_intro_text(video_id: str, seconds: int) -> str:
    tmpdir = tempfile.mkdtemp()
    out    = os.path.join(tmpdir, f"{video_id}.webm")
    try:
        subprocess.run([
            "yt-dlp","-f","bestaudio",
            "--download-sections", f"*00:00:00-00:00:{seconds:02d}",
            "-o", out, f"https://youtu.be/{video_id}"
        ], check=True)
        with open(out, "rb") as f:
            return openai_cli.audio.transcriptions.create(model="whisper-1", file=f).text
    except:
        return ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── Sidebar inputs ──
content_type = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤3 min)"])
num_matches  = st.sidebar.number_input("Results to show", 1, 10, 5)

# ── Load & filter your uploads ──
with st.spinner("Loading your uploads…"):
    my_ids = fetch_my_videos(CHANNEL_ID)
df_all = fetch_video_details(my_ids)
if df_all.empty:
    st.error("No uploads found."); st.stop()

want_short = content_type.startswith("Shorts")
df = df_all[df_all["type"] == ("Short" if want_short else "Long-Form")]
if df.empty:
    st.warning("No videos of that type."); st.stop()

# ── 1) Select your video ──
st.subheader("1) Select one of your videos")
sel = st.selectbox("Your videos", df["title"].tolist())
src = df[df["title"] == sel].iloc[0]
# keep thumbnail at a fixed width
st.image(src["thumb"], width=300, caption=sel)
st.caption(f"{src['channel']} · {src['uploadDate']} · {format_views(src['views'])} views")

# ── 2) Enter primary keyword ──
st.subheader("2) Enter a primary keyword (mandatory)")
pk = st.text_input("Keyword")
if not pk:
    st.info("Please enter a keyword."); st.stop()

# ── Precompute embeddings & visuals ──
emb_src  = get_embedding(src["title"])
text_src = extract_text_via_vision(src["thumb"])
img      = Image.open(requests.get(src["thumb"], stream=True).raw)\
               .convert("RGB").resize((256,256))
hist_src = img.histogram(); total = sum(hist_src)
def hist_sim(url: str) -> float:
    img2  = Image.open(requests.get(url, stream=True).raw)\
              .convert("RGB").resize((256,256))
    h2    = img2.histogram()
    inter = sum(min(hist_src[i], h2[i]) for i in range(len(h2)))
    return inter/total*100.0

# ── Helper: render cards ──
def render_cards(records, metrics):
    per_row = 4
    for idx, r in enumerate(records):
        if idx % per_row == 0:
            cols = st.columns(per_row)
        c = cols[idx % per_row]
        with c:
            st.image(r["thumb"], use_container_width=True)
            st.markdown(f"**[{r['title']}](https://youtu.be/{r['videoId']})**")
            st.caption(f"{r['channel']} · {r['uploadDate']} · {format_views(r['views'])}")
            for m in metrics:
                label = m.replace("_pct","%").replace("_"," ").title()
                st.markdown(f"- **{label}**: {r[m]:.1f}%")

# ── 3) Run matches ──
if st.button("3) Run Title, Thumbnail & Intro Match"):
    def yt_search(q: str, dur: str = None):
        params = dict(part="snippet", q=q, type="video",
                      order="viewCount", maxResults=50, key=YT_KEY)
        if dur:
            params["videoDuration"] = dur
        return requests.get(
            "https://youtube.googleapis.com/youtube/v3/search",
            params=params
        ).json().get("items", [])

    # gather candidates
    sem_all    = yt_search(src["title"])
    sem_sh     = yt_search(src["title"], "short")
    key_all    = yt_search(pk)
    key_sh     = yt_search(pk, "short")
    cand_sem   = [i["id"]["videoId"] for i in (sem_all+sem_sh)
                  if i["snippet"]["channelId"] in ALLOWED_CHANNELS]
    cand_key   = [i["id"]["videoId"] for i in (key_all+key_sh)
                  if i["snippet"]["channelId"] in ALLOWED_CHANNELS]
    combined   = list(dict.fromkeys(cand_sem + cand_key))
    if not combined:
        st.warning("No matches found."); st.stop()

    df_cand = fetch_video_details(combined)

    # compute metrics
    df_cand["sem_pct"]      = df_cand["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
    df_cand["key_pct"]      = df_cand["title"].map(lambda t: fuzz.ratio(pk, t))
    df_cand["combined_pct"] = df_cand[["sem_pct","key_pct"]].max(axis=1)
    df_cand["text_pct"]     = df_cand["thumb"].map(lambda u: fuzz.ratio(text_src, extract_text_via_vision(u)))
    df_cand["visual_pct"]   = df_cand["thumb"].map(hist_sim)

    secs  = 20 if want_short else 60
    intro = get_intro_text(src["videoId"], secs)
    if intro:
        df_cand["intro_title_pct"]    = df_cand["title"].map(lambda t: fuzz.ratio(intro, t))
        df_cand["intro_thumb_pct"]    = df_cand["thumb"].map(lambda u: fuzz.ratio(intro, extract_text_via_vision(u)))
        df_cand["intro_combined_pct"] = df_cand[["intro_title_pct","intro_thumb_pct"]].max(axis=1)
    else:
        df_cand["intro_title_pct"] = df_cand["intro_thumb_pct"] = df_cand["intro_combined_pct"] = 0

    # ── Table 1: Title Matches (combined_pct > 75%) ──
    st.subheader("Table 1 – Title Matches (Combined % > 75%)")
    t1l, t1s = st.tabs(["Long-Form","Shorts"])
    for tab, vt in ((t1l,"Long-Form"),(t1s,"Short")):
        with tab:
            subset = df_cand[
                (df_cand["type"]==vt) &
                (df_cand["combined_pct"]>75)
            ]
            topn = subset.nlargest(num_matches, "combined_pct")
            render_cards(topn.to_dict("records"),
                         ["sem_pct","key_pct","combined_pct"])

    # ── Table 2: Thumbnail Matches (visual_pct > 60%) ──
    st.subheader("Table 2 – Thumbnail Matches (Visual % > 60%)")
    t2l, t2s = st.tabs(["Long-Form","Shorts"])
    for tab, vt in ((t2l,"Long-Form"),(t2s,"Short")):
        with tab:
            subset = df_cand[
                (df_cand["type"]==vt) &
                (df_cand["visual_pct"]>60)
            ]
            topn = subset.nlargest(num_matches, "visual_pct")
            render_cards(topn.to_dict("records"),
                         ["text_pct","visual_pct"])

    # ── Table 3: Intro Text Matches (intro_combined_pct > 10%) ──
    st.subheader("Table 3 – Intro Text Matches (Intro Combined % > 10%)")
    if not intro:
        st.warning("No intro transcript available.")
    else:
        t3l, t3s = st.tabs(["Long-Form","Shorts"])
        for tab, vt in ((t3l,"Long-Form"),(t3s,"Short")):
            with tab:
                subset = df_cand[
                    (df_cand["type"]==vt) &
                    (df_cand["intro_combined_pct"]>10)
                ]
                topn = subset.nlargest(num_matches, "intro_combined_pct")
                render_cards(topn.to_dict("records"),
                             ["intro_title_pct","intro_thumb_pct","intro_combined_pct"])
