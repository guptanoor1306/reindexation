import re
import requests
import streamlit as st
import streamlit.components.v1 as components
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
        ids += [i["id"]["videoId"] for i in res.get("items",[])]
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
                "channelId":  v["snippet"]["channelId"],
                "channel":    v["snippet"]["channelTitle"],
                "uploadDate": datetime.fromisoformat(pub.rstrip("Z")).date().isoformat(),
                "thumb":      v["snippet"]["thumbnails"]["high"]["url"],
                "views":      int(v["statistics"].get("viewCount",0)),
                "type":       "Short" if sec <= 180 else "Long-Form"
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

def get_intro_text(video_id: str, seconds: int) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text, total = "", 0
        for seg in transcript:
            if total >= seconds:
                break
            text += " " + seg["text"]
            total += seg["duration"]
        return text.strip()
    except:
        return ""

# ── Sidebar ──
channel_id   = st.sidebar.text_input("Your Channel ID")
content_type = st.sidebar.selectbox("Filter by:", ["Long-Form (>3 min)", "Shorts (≤ 3 min)"])
num_matches  = st.sidebar.number_input("Results to show", 1, 10, 5)

if not channel_id:
    st.info("Enter your YouTube Channel ID."); st.stop()

with st.spinner("Loading your uploads…"):
    my_ids = fetch_my_videos(channel_id)
if not my_ids:
    st.error("No videos found."); st.stop()

df_all     = fetch_video_details(my_ids)
want_short = content_type.startswith("Shorts")
df         = df_all[df_all["type"] == ("Short" if want_short else "Long-Form")]
if df.empty:
    st.warning(f"No {content_type} found."); st.stop()

# ── Main: select & display your video ──
st.subheader("1) Select one of your videos")
sel = st.selectbox("Your videos", df["title"].tolist())
src = df[df["title"] == sel].iloc[0]

st.image(src["thumb"], caption=f"▶️ {sel}", width=300)
st.markdown(
    f"**Channel:** {src['channel']}   "
    f"**Uploaded:** {src['uploadDate']}   "
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
def hist_sim(u: str) -> float:
    i = Image.open(requests.get(u, stream=True).raw)\
             .convert("RGB").resize((256,256))
    h = i.histogram()
    inter = sum(min(hist_src[j], h[j]) for j in range(len(h)))
    return inter/total*100

if st.button("3) Run Title, Thumbnail & Intro Match"):
    # ── per-channel semantic & keyword search ──
    cand_sem, cand_key = [], []
    for ch in ALLOWED_CHANNELS:
        sem_resp = youtube.search().list(
            part="snippet", channelId=ch, q=src["title"],
            type="video", order="viewCount", maxResults=5
        ).execute()
        cand_sem += [item["id"]["videoId"] for item in sem_resp.get("items", [])]

        key_resp = youtube.search().list(
            part="snippet", channelId=ch, q=pk,
            type="video", order="viewCount", maxResults=5
        ).execute()
        cand_key += [item["id"]["videoId"] for item in key_resp.get("items", [])]

    combined = list(dict.fromkeys(cand_sem + cand_key))
    if not combined:
        st.warning("No matches found."); st.stop()

    df_cand = fetch_video_details(combined)

    # ── Table 1: Title Matches ──
    df_cand["sem"]   = df_cand["title"].map(lambda t: cosine_sim(emb_src, get_embedding(t)))
    df_cand["key"]   = df_cand["title"].map(lambda t: fuzz.ratio(pk, t))
    df_cand["score"] = df_cand[["sem","key"]].max(axis=1)
    df_cand.sort_values("score", ascending=False, inplace=True)
    top1 = df_cand.head(num_matches)

    html1 = """
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr>
          <th align="left">Title</th><th align="left">Channel</th>
          <th align="left">Uploaded</th><th align="right">Views</th>
          <th align="right">Sem %</th><th align="right">Key %</th>
          <th align="right">Combined %</th>
        </tr>
      </thead><tbody>
    """
    for r in top1.itertuples():
        url = f"https://youtu.be/{r.videoId}"
        html1 += f"""
        <tr>
          <td><a href="{url}" target="_blank">{r.title}</a></td>
          <td>{r.channel}</td>
          <td>{r.uploadDate}</td>
          <td style="text-align:right">{format_views(r.views)}</td>
          <td style="text-align:right">{r.sem:.1f}%</td>
          <td style="text-align:right">{r.key:.1f}%</td>
          <td style="text-align:right">{r.score:.1f}%</td>
        </tr>
        """
    html1 += "</tbody></table>"
    st.subheader("Table 1 – Title Matches")
    components.html(html1, height=50 + 40 * len(top1), scrolling=True)

    # ── Table 2: Thumbnail Matches ──
    df_cand["text"] = df_cand["thumb"].map(lambda u: fuzz.ratio(text_src, extract_text_via_vision(u)))
    df_cand["hist"] = df_cand["thumb"].map(hist_sim)
    df2 = df_cand[(df_cand["text"]>0)|(df_cand["hist"]>0)].sort_values(
        ["hist","text"], ascending=[False,False]
    ).head(num_matches)

    html2 = """
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr>
          <th align="center">Thumbnail</th><th align="left">Title</th>
          <th align="left">Channel</th><th align="left">Uploaded</th>
          <th align="right">Views</th><th align="right">Text %</th>
          <th align="right">Visual %</th>
        </tr>
      </thead><tbody>
    """
    for r in df2.itertuples():
        url = f"https://youtu.be/{r.videoId}"
        html2 += f"""
        <tr>
          <td style="text-align:center"><img src="{r.thumb}" width="120"></td>
          <td><a href="{url}" target="_blank">{r.title}</a></td>
          <td>{r.channel}</td>
          <td>{r.uploadDate}</td>
          <td style="text-align:right">{format_views(r.views)}</td>
          <td style="text-align:right">{r.text:.1f}%</td>
          <td style="text-align:right">{r.hist:.1f}%</td>
        </tr>
        """
    html2 += "</tbody></table>"
    st.subheader("Table 2 – Thumbnail Matches")
    components.html(html2, height=80 + 50 * len(df2), scrolling=True)

    # ── Table 3: Intro Text Matches ──
    secs = 20 if want_short else 60
    intro = get_intro_text(src["videoId"], secs)
    if not intro:
        st.warning("No transcript available for the intro segment.")
    else:
        df_cand["intro_title_sim"] = df_cand["title"].map(lambda t: fuzz.ratio(intro, t))
        df_cand["thumb_text"]      = df_cand["thumb"].map(extract_text_via_vision)
        df_cand["intro_thumb_sim"] = df_cand["thumb_text"].map(lambda x: fuzz.ratio(intro, x))
        df_cand["intro_score"]     = df_cand[["intro_title_sim","intro_thumb_sim"]].max(axis=1)
        top3 = df_cand.sort_values("intro_score", ascending=False).head(num_matches)

        html3 = """
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr>
              <th align="left">Title</th><th align="left">Channel</th>
              <th align="left">Uploaded</th><th align="right">Views</th>
              <th align="right">Intro→Title %</th>
              <th align="right">Intro→ThumbText %</th>
              <th align="right">Combined %</th>
            </tr>
          </thead><tbody>
        """
        for r in top3.itertuples():
            url = f"https://youtu.be/{r.videoId}"
            html3 += f"""
            <tr>
              <td><a href="{url}" target="_blank">{r.title}</a></td>
              <td>{r.channel}</td>
              <td>{r.uploadDate}</td>
              <td style="text-align:right">{format_views(r.views)}</td>
              <td style="text-align:right">{r.intro_title_sim:.1f}%</td>
              <td style="text-align:right">{r.intro_thumb_sim:.1f}%</td>
              <td style="text-align:right">{r.intro_score:.1f}%</td>
            </tr>
            """
        html3 += "</tbody></table>"
        st.subheader("Table 3 – Intro Text Matches")
        components.html(html3, height=50 + 40 * len(top3), scrolling=True)
