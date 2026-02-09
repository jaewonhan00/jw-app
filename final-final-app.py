"""
실행 방법:
  pip install streamlit requests openai python-dotenv
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd


PERIOD_DAYS = 28
PERIOD_WEEKS = 4
MIN_PER_MEAL = 5000
MIN_SNACK_DAILY = 1000
RADIUS_M = 2000
# CHANGED: tolerance/retry constants for plan validation and generation loop
TOLERANCE_WON = 50_000
MAX_RETRIES = 3

# CHANGED: keyword minimum guarantee
MIN_KEYWORDS_PER_TYPE = 8
FALLBACK_RESTAURANT_KEYWORDS = ["맛집", "한식", "백반", "국밥", "김밥", "분식", "중식", "덮밥", "돈까스", "칼국수"]
FALLBACK_CAFE_KEYWORDS = ["카페", "커피", "디저트", "베이커리", "케이크", "크로플", "도넛", "마카롱", "빙수", "빵집"]

CATEGORY_KEYS = [
    "housing",
    "eating_out",
    "groceries",
    "snack",
    "nightlife",
    "transport",
    "shopping",
    "culture",
    "etc",
]

PRIORITY_CHOICES = ["식비", "간식비", "유흥비", "쇼핑비", "문화생활비", "교통비"]
PRIORITY_MAP = {
    "식비": "eating_out",
    "간식비": "snack",
    "유흥비": "nightlife",
    "쇼핑비": "shopping",
    "문화생활비": "culture",
    "교통비": "transport",
}


def load_env_keys() -> dict[str, str]:
    load_dotenv(override=False)
    return {
        "kakao": os.getenv("KAKAO_REST_API_KEY", ""),
        "naver_id": os.getenv("NAVER_CLIENT_ID", ""),
        "naver_secret": os.getenv("NAVER_CLIENT_SECRET", ""),
    }


def mw_to_won(mw: int) -> int:
    return int(mw) * 10000


def compute_min_possible_total(
    target_total_spending_won: int,
    current_housing_won: int,
    n1_eating_out_meals_per_day: int,
    floors_won: dict[str, int],
) -> dict[str, int]:
    snack_month_min = PERIOD_DAYS * MIN_SNACK_DAILY
    eating_out_month_min = PERIOD_DAYS * n1_eating_out_meals_per_day * MIN_PER_MEAL

    floor_min_month = {k: max(0, floors_won.get(k, 0)) for k in CATEGORY_KEYS}
    floor_min_month["housing"] = current_housing_won

    snack_min_month = max(floor_min_month.get("snack", 0), snack_month_min)
    eating_out_min_month = max(floor_min_month.get("eating_out", 0), eating_out_month_min)

    min_possible_total_won = (
        current_housing_won
        + eating_out_min_month
        + snack_min_month
        + floor_min_month.get("groceries", 0)
        + floor_min_month.get("nightlife", 0)
        + floor_min_month.get("transport", 0)
        + floor_min_month.get("shopping", 0)
        + floor_min_month.get("culture", 0)
        + floor_min_month.get("etc", 0)
    )
    return {
        "min_possible_total_won": min_possible_total_won,
        "gap_won": max(0, min_possible_total_won - target_total_spending_won),
    }


def validate_feasibility(
    target_total_spending_won: int,
    current_housing_won: int,
    n1_eating_out_meals_per_day: int,
    floors_won: dict[str, int],
) -> tuple[bool, int]:
    result = compute_min_possible_total(
        target_total_spending_won=target_total_spending_won,
        current_housing_won=current_housing_won,
        n1_eating_out_meals_per_day=n1_eating_out_meals_per_day,
        floors_won=floors_won,
    )
    return (target_total_spending_won >= result["min_possible_total_won"], result["gap_won"])


def _openai_text_to_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    return json.loads(text)


def openai_generate_plan(api_key: str, input_won: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a strict budgeting planner. Always return ONLY a valid JSON object, no markdown. "
        "Currency is KRW and period is fixed to 28 days and 4 weeks. "
        "Respect constraints and floors. "
        "Ensure monthly_targets sum is equal to target_total or at least within ±50,000 KRW. "
        "If possible, absorb residual difference into etc category."
    )
    user_prompt = {
        "task": "Generate monthly spending plan.",
        "requirements": {
            "period_days": 28,
            "period_weeks": 4,
            "currency": "KRW",
            "min_per_meal": MIN_PER_MEAL,
            "min_snack_daily": MIN_SNACK_DAILY,
            "must_match_target": True,
            "output_schema": {
                "meta": {"period_days": 28, "period_weeks": 4, "currency": "KRW"},
                "monthly_targets": {
                    "housing": "int",
                    "eating_out": "int",
                    "groceries": "int",
                    "snack": "int",
                    "nightlife": "int",
                    "transport": "int",
                    "shopping": "int",
                    "culture": "int",
                    "etc": "int",
                },
                "daily_budgets": {"eating_out_daily": "int", "snack_daily": "int"},
                "eating_out": {"meals_per_day": "int", "per_meal_budget": "int"},
                "weekly_budgets": {
                    "groceries_weekly": "int",
                    "nightlife_weekly": "int",
                    "culture_weekly": "int",
                },
                "checks": {
                    "target_total": "int",
                    "sum_monthly_targets": "int",
                    "constraints_ok": "bool",
                    "notes": "string",
                },
            },
        },
        "input": input_won,
    }

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    return _openai_text_to_json(resp.output_text)


# CHANGED: tolerance-based plan verification

def verify_plan(plan: dict[str, Any], input_won: dict[str, Any]) -> tuple[bool, str, int]:
    monthly_targets = plan.get("monthly_targets", {})
    if any(k not in monthly_targets for k in CATEGORY_KEYS):
        return False, "필수 월별 항목이 누락되었습니다.", -1

    target_total = input_won["target_total_spending_won"]
    sum_monthly = int(sum(int(monthly_targets[k]) for k in CATEGORY_KEYS))
    diff = abs(sum_monthly - target_total)
    if diff > TOLERANCE_WON:
        return (
            False,
            f"월별 항목 합계와 절약 목표의 오차가 허용범위({TOLERANCE_WON:,}원)를 초과했습니다. (오차: {diff:,}원)",
            diff,
        )

    per_meal_budget = int(plan.get("eating_out", {}).get("per_meal_budget", 0))
    snack_daily = int(plan.get("daily_budgets", {}).get("snack_daily", 0))
    if per_meal_budget < MIN_PER_MEAL:
        return False, f"한 끼당 외식비는 최소 {MIN_PER_MEAL}원이어야 합니다.", diff
    if snack_daily < MIN_SNACK_DAILY:
        return False, f"하루 간식비는 최소 {MIN_SNACK_DAILY}원이어야 합니다.", diff

    floors = input_won["floors_won"]
    for k in ["groceries", "nightlife", "transport", "shopping", "culture", "etc", "snack", "eating_out"]:
        if int(monthly_targets.get(k, 0)) < int(floors.get(k, 0)):
            return False, f"{k} 항목이 하한선을 만족하지 못했습니다.", diff

    if int(monthly_targets.get("housing", 0)) < int(input_won["current_categories_won"]["housing"]):
        return False, "거주비가 현재 거주비보다 낮게 설정되었습니다.", diff

    return True, "검증 성공", diff


def choose_price_band_eating_out(per_meal_budget: int) -> dict[str, Any]:
    if per_meal_budget < 8000:
        return {"label": "초저가", "min": 0, "max": 8000}
    if per_meal_budget < 13000:
        return {"label": "실속", "min": 8000, "max": 13000}
    if per_meal_budget < 20000:
        return {"label": "보통", "min": 13000, "max": 20000}
    return {"label": "여유", "min": 20000, "max": 1000000}


def choose_price_band_snack(snack_daily_budget: int) -> dict[str, Any]:
    if snack_daily_budget < 4000:
        return {"label": "저가", "min": 0, "max": 4000}
    if snack_daily_budget < 8000:
        return {"label": "중가", "min": 4000, "max": 8000}
    return {"label": "고가", "min": 8000, "max": 1000000}


def openai_infer_keywords(api_key: str, location_label: str, eat_band: dict[str, Any], snack_band: dict[str, Any]) -> dict[str, list[str]]:
    client = OpenAI(api_key=api_key)
    prompt = {
        "task": "Suggest Korean local search keywords for restaurant and cafe recommendation.",
        "location_label": location_label,
        "restaurant_price_band": eat_band,
        "cafe_price_band": snack_band,
        "format": {"restaurant_keywords": ["string"], "cafe_keywords": ["string"]},
        "rules": "JSON only. Provide 5 keywords each, Korean text.",
    }
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            temperature=0.3,
        )
        data = _openai_text_to_json(resp.output_text)
        # CHANGED: keyword minimum guarantee
        rk_raw = [str(x).strip() for x in data.get("restaurant_keywords", []) if str(x).strip()]
        ck_raw = [str(x).strip() for x in data.get("cafe_keywords", []) if str(x).strip()]
    except Exception:
        rk_raw = []
        ck_raw = []

    def _merge_minimum(primary: list[str], fallback: list[str], min_count: int) -> list[str]:
        merged = []
        seen = set()
        for kw in primary + fallback:
            if kw in seen:
                continue
            seen.add(kw)
            merged.append(kw)
            if len(merged) >= min_count:
                break
        return merged

    rk = _merge_minimum(rk_raw, FALLBACK_RESTAURANT_KEYWORDS, MIN_KEYWORDS_PER_TYPE)
    ck = _merge_minimum(ck_raw, FALLBACK_CAFE_KEYWORDS, MIN_KEYWORDS_PER_TYPE)
    return {"restaurant_keywords": rk, "cafe_keywords": ck}


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address_kakao(address: str, kakao_key: str) -> dict[str, Any] | None:
    if not address:
        return None
    try:
        resp = requests.get(
            "https://dapi.kakao.com/v2/local/search/address.json",
            headers={"Authorization": f"KakaoAK {kakao_key}"},
            params={"query": address},
            timeout=8,
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if not docs:
            return None
        d = docs[0]
        return {"lat": float(d["y"]), "lon": float(d["x"])}
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def search_places_kakao(
    keyword: str,
    lat: float,
    lon: float,
    kakao_key: str,
    category_group_code: str,
    size: int = 15,
    sort: str = "accuracy",
) -> list[dict[str, Any]]:
    try:
        resp = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers={"Authorization": f"KakaoAK {kakao_key}"},
            params={
                "query": keyword,
                "y": lat,
                "x": lon,
                "radius": RADIUS_M,
                "sort": sort,
                "category_group_code": category_group_code,
                "size": min(size, 30),
                "page": 1,
            },
            timeout=8,
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        results = []
        for d in docs:
            results.append(
                {
                    "id": d.get("id"),
                    "name": d.get("place_name"),
                    "category": d.get("category_name", ""),
                    "address": d.get("road_address_name") or d.get("address_name"),
                    "lat": float(d.get("y", 0)),
                    "lon": float(d.get("x", 0)),
                    "distance_kakao": int(d.get("distance", 0) or 0),
                }
            )
        return results
    except Exception:
        return []


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def normalize_linear_distance_to_1_100(distance_m: float) -> int:
    if distance_m <= 0:
        return 100
    if distance_m >= RADIUS_M:
        return 1
    score = 100 - (distance_m / RADIUS_M) * 99
    return int(round(max(1, min(100, score))))


@st.cache_data(show_spinner=False, ttl=1800)
def naver_blog_total(query: str, naver_id: str, naver_secret: str) -> int:
    if not naver_id or not naver_secret:
        return 0
    try:
        resp = requests.get(
            "https://openapi.naver.com/v1/search/blog.json",
            headers={
                "X-Naver-Client-Id": naver_id,
                "X-Naver-Client-Secret": naver_secret,
            },
            params={"query": query, "display": 1, "start": 1, "sort": "sim"},
            timeout=8,
        )
        resp.raise_for_status()
        return int(resp.json().get("total", 0))
    except Exception:
        return 0


def normalize_minmax_to_1_100(values: list[int]) -> list[int]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return [50 for _ in values]
    out = []
    for v in values:
        score = 1 + (v - vmin) * (99 / (vmax - vmin))
        out.append(int(round(score)))
    return out


def rank_places(
    origin: dict[str, float],
    candidates: list[dict[str, Any]],
    naver_id: str,
    naver_secret: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    for c in candidates:
        dist = haversine_distance_m(origin["lat"], origin["lon"], c["lat"], c["lon"])
        c["distance_m"] = int(round(dist))
        c["p1"] = normalize_linear_distance_to_1_100(dist)
        c["blog_total"] = naver_blog_total(c["name"], naver_id, naver_secret)

    p2_list = normalize_minmax_to_1_100([c["blog_total"] for c in candidates])
    for idx, c in enumerate(candidates):
        c["p2"] = p2_list[idx]
        c["score"] = round(0.5 * c["p1"] + 0.5 * c["p2"], 2)
        c["reason"] = f"예산 구간 적합 / 거리 점수 {c['p1']} / 블로그 언급 점수 {c['p2']}"

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def recommend_for_locations(
    api_key: str,
    env_keys: dict[str, str],
    plan: dict[str, Any],
    locations: list[dict[str, str]],
    exclusion_by_location: dict[str, dict[str, set[str]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    per_meal = int(plan["eating_out"]["per_meal_budget"])
    snack_daily = int(plan["daily_budgets"]["snack_daily"])
    eat_band = choose_price_band_eating_out(per_meal)
    snack_band = choose_price_band_snack(snack_daily)

    kakao_key = env_keys.get("kakao", "")
    naver_id = env_keys.get("naver_id", "")
    naver_secret = env_keys.get("naver_secret", "")

    if not kakao_key:
        raise RuntimeError("Kakao REST API 키를 .env에서 찾지 못했습니다.")

    for loc in locations:
        label = loc["label"]
        address = loc["address"]
        origin = geocode_address_kakao(address, kakao_key)
        if not origin:
            output[label] = {"restaurants": [], "cafes": []}
            continue

        kws = openai_infer_keywords(api_key, label, eat_band, snack_band)

        def _collect_candidates(
            keywords: list[str],
            category_code: str,
            exclusion_set: set[str],
            fallback_keywords: list[str],
        ) -> list[dict[str, Any]]:
            # 1차 검색 (정확도)
            all_items: list[dict[str, Any]] = []
            for kw in keywords:
                all_items.extend(
                    search_places_kakao(kw, origin["lat"], origin["lon"], kakao_key, category_code, 15, "accuracy")
                )

            def _dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
                seen_local = set()
                unique: list[dict[str, Any]] = []
                for item in items:
                    item_id = item.get("id")
                    if not item_id or item_id in seen_local or item_id in exclusion_set:
                        continue
                    seen_local.add(item_id)
                    unique.append(item)
                return unique

            uniq_items = _dedup(all_items)

            # CHANGED: fallback search
            if len(uniq_items) == 0:
                fallback_items: list[dict[str, Any]] = []
                for kw in fallback_keywords:
                    fallback_items.extend(
                        search_places_kakao(kw, origin["lat"], origin["lon"], kakao_key, category_code, 15, "accuracy")
                    )
                uniq_items = _dedup(fallback_items)

            # CHANGED: relaxed search
            if 0 < len(uniq_items) < 5:
                relaxed_queries = ["맛집", "음식점"] if category_code == "FD6" else ["카페", "디저트"]
                relaxed_items: list[dict[str, Any]] = list(uniq_items)
                for kw in relaxed_queries:
                    relaxed_items.extend(
                        search_places_kakao(kw, origin["lat"], origin["lon"], kakao_key, category_code, 30, "distance")
                    )
                uniq_items = _dedup(relaxed_items)

            return uniq_items

        rest_ex = exclusion_by_location.get(label, {}).get("restaurants", set())
        cafe_ex = exclusion_by_location.get(label, {}).get("cafes", set())

        uniq_rest = _collect_candidates(kws["restaurant_keywords"], "FD6", rest_ex, FALLBACK_RESTAURANT_KEYWORDS)
        uniq_cafe = _collect_candidates(kws["cafe_keywords"], "CE7", cafe_ex, FALLBACK_CAFE_KEYWORDS)

        ranked_rest = rank_places(origin, uniq_rest, naver_id, naver_secret)[:5]
        ranked_cafe = rank_places(origin, uniq_cafe, naver_id, naver_secret)[:5]

        # CHANGED: no-result message plumbing
        output[label] = {
            "restaurants": ranked_rest,
            "cafes": ranked_cafe,
            "messages": {
                "restaurants": "조건에 맞는 가게를 찾지 못했습니다." if len(ranked_rest) == 0 else None,
                "cafes": "조건에 맞는 가게를 찾지 못했습니다." if len(ranked_cafe) == 0 else None,
            },
        }
    return output


def _render_place_list(items: list[dict[str, Any]]) -> None:
    if not items:
        st.caption("추천 결과가 없습니다.")
        return
    for it in items:
        st.markdown(
            f"- **{it['name']}**  \\n"
            f"  카테고리: {it['category']}  \\n"
            f"  거리: {it['distance_m']}m  \\n"
            f"  최종점수: {it['score']}  \\n"
            f"  추천 이유: {it['reason']}"
        )


def _build_input_data() -> dict[str, Any]:
    st.header("소비 플랜 생성을 위해 다음의 정보를 입력해주세요.")

    st.subheader("현재 월 수입")
    monthly_income_mw = st.number_input("현재 월 수입", min_value=0, step=1, value=300)
    st.caption("단위: 10,000원")

    st.subheader("현재 월 소비액")
    current_total_spending_mw = st.number_input("현재 월 총 소비액", min_value=0, step=1, value=220)
    st.caption("단위: 10,000원")

    left, right = st.columns(2)
    with left:
        food_total_mw = st.number_input("식비", min_value=0, step=1, value=70)
        st.caption("단위: 10,000원")
        n1 = st.selectbox("하루 3끼 중 외식이 차지하는 끼니 횟수", [0, 1, 2, 3], index=1)
        eating_out_mw = st.number_input("식비 중 외식비", min_value=0, step=1, value=45)
        st.caption("배달 음식 포함")
        st.caption("단위: 10,000원")
        groceries_mw = st.number_input("식비 중 식재료비", min_value=0, step=1, value=25)
        st.caption("외식비를 제외한 식비")
        st.caption("단위: 10,000원")

    with right:
        snack_mw = st.number_input("간식비", min_value=0, step=1, value=15)
        st.caption("카페, 간식, 음료 등")
        st.caption("단위: 10,000원")
        nightlife_mw = st.number_input("유흥비", min_value=0, step=1, value=15)
        st.caption("술자리 관련")
        st.caption("단위: 10,000원")
        shopping_mw = st.number_input("쇼핑비", min_value=0, step=1, value=15)
        st.caption("의류, 잡화, 생활용품 등")
        st.caption("단위: 10,000원")
        culture_mw = st.number_input("문화생활비", min_value=0, step=1, value=20)
        st.caption("영화, 전시, 게임, 여행, 덕질 등")
        st.caption("단위: 10,000원")
        transport_mw = st.number_input("교통비", min_value=0, step=1, value=12)
        st.caption("단위: 10,000원")
        housing_mw = st.number_input("거주비", min_value=0, step=1, value=50)
        st.caption("집세, 관리비, 공과금 포함")
        st.caption("단위: 10,000원")
        etc_mw = st.number_input("기타", min_value=0, step=1, value=8)
        st.caption("단위: 10,000원")

    st.subheader("절약 목표(원하는 월 총 소비액)")
    target_total_spending_mw = st.number_input("절약 목표", min_value=0, step=1, value=180)
    st.caption("단위: 10,000원")

    st.subheader("가장 절약하고 싶은 항목 순위")
    selected = []
    priority_out = []
    for i in range(1, 6):
        options = ["선택 안 함"] + [x for x in PRIORITY_CHOICES if x not in selected]
        pick = st.selectbox(f"rank_{i}", options, key=f"rank_{i}")
        if pick != "선택 안 함":
            selected.append(pick)
            priority_out.append(PRIORITY_MAP[pick])

    st.subheader("하한선 설정")
    st.write(
        "절약에 한계가 있는 항목에 대해 더이상 줄일 수 없는 소비액 하한선을 설정해주세요.(ex. 교통비 10만 원) "
        "하한선이 필요한 항목이 없다면 입력하지 않아도 됩니다."
    )
    st.caption("월 기준 하한선")

    floors_mw = {}
    floors_mw["eating_out"] = st.number_input("하한선 - 외식비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["groceries"] = st.number_input("하한선 - 식재료비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["snack"] = st.number_input("하한선 - 간식비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["nightlife"] = st.number_input("하한선 - 유흥비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["transport"] = st.number_input("하한선 - 교통비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["shopping"] = st.number_input("하한선 - 쇼핑비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["culture"] = st.number_input("하한선 - 문화생활비", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")
    floors_mw["etc"] = st.number_input("하한선 - 기타", min_value=0, step=1, value=0)
    st.caption("단위: 10,000원")

    st.subheader("위치 설정")
    st.write(
        "당신의 거주지, 직장 위치, 자주 다니는 거리의 위치 등을 입력하면 생성한 소비 플랜에 걸맞는 가게를 추천해드립니다."
    )

    if "ui_location_count" not in st.session_state:
        st.session_state["ui_location_count"] = 3

    default_labels = ["거주지", "직장", "자주 다니는 골목"]
    locations = []
    for i in range(st.session_state["ui_location_count"]):
        label_default = default_labels[i] if i < len(default_labels) else f"위치 {i+1}"
        c1, c2 = st.columns(2)
        with c1:
            lbl = st.text_input(f"위치 라벨 {i+1}", value=label_default, key=f"loc_label_{i}")
        with c2:
            addr = st.text_input(f"주소 {i+1}", value="", key=f"loc_addr_{i}")
        if addr.strip():
            locations.append({"label": lbl.strip() or f"위치 {i+1}", "address": addr.strip()})

    if st.button("추가"):
        st.session_state["ui_location_count"] += 1
        st.rerun()

    return {
        "monthly_income_mw": int(monthly_income_mw),
        "current_total_spending_mw": int(current_total_spending_mw),
        "current_categories_mw": {
            "housing": int(housing_mw),
            "food_total": int(food_total_mw),
            "eating_out": int(eating_out_mw),
            "groceries": int(groceries_mw),
            "snack": int(snack_mw),
            "nightlife": int(nightlife_mw),
            "transport": int(transport_mw),
            "shopping": int(shopping_mw),
            "culture": int(culture_mw),
            "etc": int(etc_mw),
        },
        "n1_eating_out_meals_per_day": int(n1),
        "target_total_spending_mw": int(target_total_spending_mw),
        "saving_priority": priority_out,
        "floors_mw": {k: int(v) for k, v in floors_mw.items() if int(v) > 0},
        "locations": locations,
    }


def _to_input_won(input_data: dict[str, Any]) -> dict[str, Any]:
    current_categories_won = {
        k: mw_to_won(v) for k, v in input_data["current_categories_mw"].items() if k != "food_total"
    }
    floors_won = {k: mw_to_won(v) for k, v in input_data.get("floors_mw", {}).items()}

    return {
        "monthly_income_won": mw_to_won(input_data["monthly_income_mw"]),
        "current_total_spending_won": mw_to_won(input_data["current_total_spending_mw"]),
        "current_categories_won": current_categories_won,
        "n1_eating_out_meals_per_day": input_data["n1_eating_out_meals_per_day"],
        "target_total_spending_won": mw_to_won(input_data["target_total_spending_mw"]),
        "saving_priority": input_data["saving_priority"],
        "floors_won": floors_won,
        "locations": input_data["locations"],
    }


def _render_plan(plan: dict[str, Any]) -> None:
    st.header("한 달 플랜")

    snack_daily = int(plan["daily_budgets"]["snack_daily"])
    per_meal = int(plan["eating_out"]["per_meal_budget"])

    rows = []
    for w in range(1, 5):
        row = {f"Day{i}": f"간식: {snack_daily:,}원\n외식(1끼): {per_meal:,}원" for i in range(1, 8)}
        row["주 단위 플랜"] = (
            f"식재료비(주): {int(plan['weekly_budgets']['groceries_weekly']):,}원\n"
            f"유흥비(주): {int(plan['weekly_budgets']['nightlife_weekly']):,}원\n"
            f"문화생활(주): {int(plan['weekly_budgets']['culture_weekly']):,}원"
        )
        row["Week"] = f"{w}주차"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Week")
    st.table(df)

    monthly = plan["monthly_targets"]
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"월 교통비: {int(monthly['transport']):,}원")
    with c2:
        st.info(f"월 쇼핑비: {int(monthly['shopping']):,}원")

    # CHANGED: show target vs sum with tolerance-aware diff
    monthly_sum = int(sum(int(monthly.get(k, 0)) for k in CATEGORY_KEYS))
    target_total = int(plan.get("checks", {}).get("target_total", monthly_sum))
    signed_diff = monthly_sum - target_total
    st.caption(
        f"월별 항목 합계: {monthly_sum:,}원 / 절약 목표: {target_total:,}원 (오차: {signed_diff:+,}원)"
    )
    if abs(signed_diff) > TOLERANCE_WON:
        st.warning("월 총 소비액과 항목 합계의 오차가 허용 범위를 초과합니다. 입력 또는 생성 결과를 다시 확인해주세요.")


def main() -> None:
    st.set_page_config(page_title="절약 플래너", layout="wide")
    st.title("절약 플래너")
    st.write(
        "절약은 좋지만, 무조건 절약하다보면 삶의 품질이 매우 저하되기 마련입니다. 여러분의 현재 소비액과 절약 목표를 넣어 "
        "그것이 실현 가능한 목표인지, 그 목표를 이루기 위해서는 한 달 동안 어떻게 살아야 하는지 알아보세요.\n\n"
        "여러분의 수입, 소비액, 절약 목표 등을 입력하면 한 달간의 소비 플랜과 해당 플랜에 맞는 가게를 추천해드립니다."
    )
    st.divider()
    st.caption("사이드바에서 OpenAI API key를 반드시 입력해주세요!")

    env_keys = load_env_keys()

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("OPENAI_API_KEY", ""))
        st.session_state["OPENAI_API_KEY"] = api_key.strip()

    input_data = _build_input_data()

    if st.button("플랜 생성"):
        if not st.session_state.get("OPENAI_API_KEY"):
            st.error("OpenAI API Key를 사이드바에 입력해주세요.")
            st.stop()

        input_won = _to_input_won(input_data)
        ok, gap_won = validate_feasibility(
            target_total_spending_won=input_won["target_total_spending_won"],
            current_housing_won=input_won["current_categories_won"]["housing"],
            n1_eating_out_meals_per_day=input_won["n1_eating_out_meals_per_day"],
            floors_won=input_won["floors_won"],
        )
        if not ok:
            n_mw = math.ceil(gap_won / 10000)
            st.error(
                f"해당 설정에 맞는 플랜을 생성할 수 없습니다. 항목별 하한선을 최소 {n_mw}만 원 더 낮게 설정하거나, "
                f"절약 목표 금액을 {n_mw}만 원 더 높게 설정해주세요."
            )
            if st.button("입력으로 돌아가기"):
                for k in ["plan", "recommendations", "excluded_ids"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
            st.stop()

        # CHANGED: retry generation when tolerance condition is not met
        for k in ["plan", "recommendations", "excluded_ids"]:
            if k in st.session_state:
                del st.session_state[k]

        plan_ready = False
        last_fail_msg = ""
        last_diff = -1

        with st.spinner("플랜 생성 중..."):
            for attempt in range(MAX_RETRIES + 1):
                try:
                    plan = openai_generate_plan(st.session_state["OPENAI_API_KEY"], input_won)
                except Exception:
                    st.error("OpenAI 플랜 생성 중 오류가 발생했습니다. API Key/네트워크를 확인 후 다시 시도해주세요.")
                    st.stop()

                verified, msg, diff = verify_plan(plan, input_won)
                if verified:
                    st.session_state["plan"] = plan
                    st.session_state["last_input_won"] = input_won
                    st.session_state["recommendations"] = None
                    st.session_state["excluded_ids"] = {}
                    st.success(f"목표 대비 오차 {diff:,}원(허용범위 {TOLERANCE_WON:,}원)")
                    plan_ready = True
                    break

                last_fail_msg = msg
                last_diff = diff

                # 오차 범위 초과일 때만 재시도
                if diff > TOLERANCE_WON and attempt < MAX_RETRIES:
                    retry_i = attempt + 1
                    st.info(f"플랜 생성 재시도 {retry_i}/{MAX_RETRIES} ... (오차 {diff:,}원)")
                    continue

                break

        if not plan_ready:
            st.error("플랜 생성에 여러 번 실패했습니다. 입력값을 조정하거나 다시 시도해주세요.")
            if last_diff >= 0:
                st.error(f"마지막 실패 사유: {last_fail_msg} (오차: {last_diff:,}원)")
            else:
                st.error(f"마지막 실패 사유: {last_fail_msg}")
            st.stop()

    if "plan" in st.session_state:
        _render_plan(st.session_state["plan"])

        if st.button("플랜에 맞는 가게 추천받기"):
            with st.spinner("추천 생성 중..."):
                try:
                    rec = recommend_for_locations(
                        api_key=st.session_state["OPENAI_API_KEY"],
                        env_keys=env_keys,
                        plan=st.session_state["plan"],
                        locations=st.session_state["last_input_won"]["locations"],
                        exclusion_by_location=st.session_state.get("excluded_ids", {}),
                    )
                    st.session_state["recommendations"] = rec
                except Exception as e:
                    st.error(f"추천 생성 실패: {e}")

    if st.session_state.get("recommendations"):
        st.subheader("추천 결과")
        for label, rec in st.session_state["recommendations"].items():
            st.markdown(f"### 위치: {label}")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("식당")
                # CHANGED: no-result message plumbing
                if rec.get("messages", {}).get("restaurants"):
                    st.warning(rec["messages"]["restaurants"])
                _render_place_list(rec["restaurants"])
            with c2:
                st.subheader("카페 및 디저트")
                # CHANGED: no-result message plumbing
                if rec.get("messages", {}).get("cafes"):
                    st.warning(rec["messages"]["cafes"])
                _render_place_list(rec["cafes"])

        if st.button("재추천받기"):
            excluded = st.session_state.get("excluded_ids", {})
            for label, rec in st.session_state["recommendations"].items():
                if label not in excluded:
                    excluded[label] = {"restaurants": set(), "cafes": set()}
                excluded[label]["restaurants"].update([x["id"] for x in rec["restaurants"] if x.get("id")])
                excluded[label]["cafes"].update([x["id"] for x in rec["cafes"] if x.get("id")])
            st.session_state["excluded_ids"] = excluded

            with st.spinner("재추천 생성 중..."):
                try:
                    st.session_state["recommendations"] = recommend_for_locations(
                        api_key=st.session_state["OPENAI_API_KEY"],
                        env_keys=env_keys,
                        plan=st.session_state["plan"],
                        locations=st.session_state["last_input_won"]["locations"],
                        exclusion_by_location=st.session_state.get("excluded_ids", {}),
                    )
                except Exception as e:
                    st.error(f"재추천 실패: {e}")


if __name__ == "__main__":
    main()
