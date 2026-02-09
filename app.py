import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# Google Calendar imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except Exception:
    Request = None
    Credentials = None
    InstalledAppFlow = None
    build = None

st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관을 체크하고 AI 코치 리포트를 받아보세요.")

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


# -----------------------------
# Sidebar: API Keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password")
    youtube_api_key = st.text_input("YouTube Data API Key", type="password")


# -----------------------------
# API helper functions
# -----------------------------
def get_weather(city: str, api_key: str) -> dict[str, Any] | None:
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "city": data.get("name", city),
            "temp": data.get("main", {}).get("temp"),
            "desc": (data.get("weather") or [{}])[0].get("description", "정보 없음"),
        }
    except Exception:
        return None


def get_dog_image() -> dict[str, str] | None:
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        image_url = data.get("message", "")
        if not image_url:
            return None

        parts = image_url.split("/breeds/")
        breed = "unknown"
        if len(parts) > 1:
            breed_token = parts[1].split("/")[0]
            breed = breed_token.replace("-", " ")

        return {"url": image_url, "breed": breed}
    except Exception:
        return None


def authenticate_google() -> Any | None:
    """Authenticate user with OAuth2 and return Google credentials."""
    if Request is None or Credentials is None or InstalledAppFlow is None:
        return None

    creds = None
    try:
        if "google_token" in st.session_state:
            creds = Credentials.from_authorized_user_info(st.session_state.google_token, SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state.google_token = creds.to_json()
            return creds

        if creds and creds.valid:
            return creds

        client_config = None
        if "google_oauth" in st.secrets:
            client_config = st.secrets["google_oauth"]

        if client_config:
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)

        creds = flow.run_local_server(port=0)
        st.session_state.google_token = creds.to_json()
        return creds
    except Exception:
        return None


def create_calendar_event(
    creds: Any,
    habit_name: str,
    selected_date: datetime.date,
    selected_time: datetime.time,
) -> dict[str, Any] | None:
    """Create a Google Calendar event for the selected habit."""
    if build is None:
        return None

    try:
        service = build("calendar", "v3", credentials=creds)
        start_dt = datetime.datetime.combine(selected_date, selected_time)
        end_dt = start_dt + datetime.timedelta(hours=1)

        event = {
            "summary": f"Habit: {habit_name}",
            "description": "AI Habit Tracker",
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Seoul"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Seoul"},
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        return created
    except Exception:
        return None


def build_music_query(weather: dict[str, Any] | None, mood: int, habits: list[str]) -> str:
    """Build music search query using weather, mood, and completed habits."""
    weather_desc = weather.get("desc", "일상") if weather else "일상"
    habit_count = len(habits)

    if mood <= 3:
        mood_keyword = "relax music"
    elif mood <= 7:
        mood_keyword = "focus music"
    else:
        mood_keyword = "energetic music"

    habit_keyword = "study" if habit_count >= 3 else "healing"
    return f"{weather_desc} {mood_keyword} {habit_keyword}"


def get_youtube_music(api_key: str, query: str) -> list[dict[str, str]] | None:
    """Search YouTube videos and return up to 3 music recommendations."""
    if not api_key or not query:
        return None

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": api_key,
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoEmbeddable": "true",
        "maxResults": 3,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        videos = []
        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            videos.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", "제목 없음"),
                    "channel": snippet.get("channelTitle", "채널 정보 없음"),
                    "thumbnail": (
                        snippet.get("thumbnails", {}).get("high", {}).get("url")
                        or snippet.get("thumbnails", {}).get("medium", {}).get("url")
                        or snippet.get("thumbnails", {}).get("default", {}).get("url")
                    ),
                }
            )

        return videos if videos else None
    except Exception:
        return None


def generate_report(
    *,
    openai_key: str,
    coach_style: str,
    habits: list[str],
    mood: int,
    weather: dict[str, Any] | None,
    dog_breed: str,
    achievement_rate: float,
) -> str:
    if not openai_key:
        return "OpenAI API Key를 먼저 입력해주세요."

    style_prompts = {
        "스파르타 코치": "당신은 엄격하고 직설적인 스파르타 코치다. 핑계 없이 실행 중심의 조언을 한다.",
        "따뜻한 멘토": "당신은 공감과 격려를 중시하는 따뜻한 멘토다. 부드럽지만 실천 가능한 조언을 준다.",
        "게임 마스터": "당신은 RPG 세계관의 게임 마스터다. 사용자의 하루를 퀘스트/레벨업 관점에서 코칭한다.",
    }

    weather_text = (
        f"{weather.get('city')} / {weather.get('temp')}°C / {weather.get('desc')}"
        if weather
        else "날씨 정보 없음"
    )

    user_content = f"""
오늘 체크한 습관: {', '.join(habits) if habits else '없음'}
기분 점수(1~10): {mood}
달성률: {achievement_rate:.0f}%
날씨: {weather_text}
강아지 품종: {dog_breed}

아래 형식으로 한국어 리포트를 작성해줘.
1) 컨디션 등급(S~D)
2) 습관 분석
3) 날씨 코멘트
4) 내일 미션
5) 오늘의 한마디
"""

    try:
        client = OpenAI(api_key=openai_key)
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": style_prompts.get(coach_style, "친절한 코치")},
                {"role": "user", "content": user_content},
            ],
        )
        return response.output_text.strip()
    except Exception as e:
        return f"리포트 생성에 실패했습니다: {e}"


# -----------------------------
# Habit check-in UI
# -----------------------------
st.subheader("✅ 오늘의 습관 체크인")

habit_options = [
    "🌅 기상 미션",
    "💧 물 마시기",
    "📚 공부/독서",
    "🏃 운동하기",
    "😴 수면",
]

c1, c2 = st.columns(2)
checked = []
for idx, label in enumerate(habit_options):
    with (c1 if idx % 2 == 0 else c2):
        if st.checkbox(label):
            checked.append(label)

mood = st.slider("😊 오늘 기분 점수", min_value=1, max_value=10, value=6)

city_list = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Suwon",
    "Ulsan",
    "Jeju",
    "Changwon",
]

col_city, col_style = st.columns(2)
with col_city:
    selected_city = st.selectbox("🌍 도시 선택", city_list)
with col_style:
    coach_style = st.radio("🧠 코치 스타일", ["스파르타 코치", "따뜻한 멘토", "게임 마스터"])


# -----------------------------
# Achievement + metrics + chart
# -----------------------------
achievement_rate = (len(checked) / len(habit_options)) * 100

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_rate:.0f}%")
m2.metric("달성 습관", f"{len(checked)} / {len(habit_options)}")
m3.metric("기분", f"{mood}/10")

if "history" not in st.session_state:
    st.session_state.history = [
        {"date": "D-6", "achievement": 40},
        {"date": "D-5", "achievement": 60},
        {"date": "D-4", "achievement": 20},
        {"date": "D-3", "achievement": 80},
        {"date": "D-2", "achievement": 50},
        {"date": "D-1", "achievement": 70},
    ]

if st.button("📌 오늘 기록 저장"):
    today_label = datetime.date.today().strftime("%m-%d")
    st.session_state.history = st.session_state.history[-6:] + [
        {"date": today_label, "achievement": round(achievement_rate)}
    ]
    st.success("오늘 기록이 저장되었습니다.")

chart_data = pd.DataFrame(st.session_state.history[-6:])
if chart_data.empty or chart_data.iloc[-1]["date"] != datetime.date.today().strftime("%m-%d"):
    chart_data = pd.concat(
        [
            chart_data,
            pd.DataFrame(
                [{"date": datetime.date.today().strftime("%m-%d"), "achievement": round(achievement_rate)}]
            ),
        ],
        ignore_index=True,
    )

st.subheader("📈 최근 7일 달성률")
st.bar_chart(chart_data.set_index("date"))


# -----------------------------
# Calendar scheduler
# -----------------------------
st.subheader("📅 Calendar")
cal_col1, cal_col2, cal_col3 = st.columns(3)
with cal_col1:
    schedule_date = st.date_input("날짜 선택", value=datetime.date.today())
with cal_col2:
    schedule_time = st.time_input("시간 선택", value=datetime.time(hour=7, minute=0))
with cal_col3:
    schedule_habit = st.selectbox("습관 선택", habit_options)

if st.button("Add to Google Calendar"):
    credentials = authenticate_google()
    if not credentials:
        st.warning("Google 인증에 실패했습니다. OAuth 설정(credential/secrets)을 확인해주세요.")
    else:
        event = create_calendar_event(credentials, schedule_habit, schedule_date, schedule_time)
        if event:
            st.success("Google Calendar에 이벤트를 추가했습니다.")
        else:
            st.warning("캘린더 이벤트 생성에 실패했습니다.")


# -----------------------------
# Result area (weather + dog + AI report)
# -----------------------------
st.subheader("🤖 AI 코치 리포트")

if "latest_weather" not in st.session_state:
    st.session_state.latest_weather = None
if "latest_report" not in st.session_state:
    st.session_state.latest_report = ""

if st.button("컨디션 리포트 생성", type="primary"):
    weather = get_weather(selected_city, weather_api_key)
    dog = get_dog_image()
    dog_breed = dog["breed"] if dog else "unknown"

    report_text = generate_report(
        openai_key=openai_api_key,
        coach_style=coach_style,
        habits=checked,
        mood=mood,
        weather=weather,
        dog_breed=dog_breed,
        achievement_rate=achievement_rate,
    )

    st.session_state.latest_weather = weather
    st.session_state.latest_report = report_text

    card1, card2 = st.columns(2)
    with card1:
        st.markdown("### 🌤️ 오늘 날씨")
        if weather:
            st.info(f"{weather['city']} · {weather['temp']}°C · {weather['desc']}")
        else:
            st.warning("날씨 정보를 가져오지 못했습니다.")

    with card2:
        st.markdown("### 🐶 랜덤 강아지")
        if dog:
            st.image(dog["url"], caption=f"Breed: {dog['breed']}", use_container_width=True)
        else:
            st.warning("강아지 이미지를 가져오지 못했습니다.")

    st.markdown("### 📝 AI 리포트")
    st.write(report_text)

    share_text = (
        f"[AI 습관 트래커]\n"
        f"도시: {selected_city}\n"
        f"달성률: {achievement_rate:.0f}% ({len(checked)}/{len(habit_options)})\n"
        f"기분: {mood}/10\n"
        f"코치 스타일: {coach_style}\n"
        f"체크 습관: {', '.join(checked) if checked else '없음'}"
    )
    st.markdown("### 📤 공유용 텍스트")
    st.code(share_text, language="text")


# -----------------------------
# Music recommendation
# -----------------------------
st.subheader("🎵 Music Recommendation")
current_weather = st.session_state.latest_weather or get_weather(selected_city, weather_api_key)
music_query = build_music_query(current_weather, mood, checked)

if st.button("음악 추천 받기"):
    videos = get_youtube_music(youtube_api_key, music_query)
    if not videos:
        st.warning("음악 추천을 가져오지 못했습니다. YouTube API 키와 네트워크를 확인해주세요.")
    else:
        st.caption(f"검색 쿼리: {music_query}")
        for video in videos:
            vc1, vc2 = st.columns([1, 2])
            with vc1:
                if video.get("thumbnail"):
                    st.image(video["thumbnail"], use_container_width=True)
            with vc2:
                st.markdown(f"**{video['title']}**")
                st.write(f"채널: {video['channel']}")
                st.video(f"https://www.youtube.com/watch?v={video['video_id']}")


with st.expander("API 안내"):
    st.markdown(
        """
- OpenAI API Key: AI 코치 리포트 생성에 사용됩니다.
- OpenWeatherMap API Key: 도시별 현재 날씨를 가져옵니다.
- YouTube Data API Key: 날씨/기분/습관 기반 음악 추천을 가져옵니다.
- Dog CEO API: 랜덤 강아지 이미지를 가져옵니다 (키 불필요).
- Google Calendar OAuth: 습관 일정을 구글 캘린더 이벤트로 저장합니다.

※ API 키는 세션 내에서만 사용되며 저장되지 않습니다.
"""
    )
