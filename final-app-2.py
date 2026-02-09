"""
실행 방법:
  pip install streamlit requests openai python-dotenv
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

PERIOD_DAYS = 28
PERIOD_WEEKS = 4
MIN_PER_MEAL = 5000
MIN_SNACK_DAILY = 1000
RADIUS_M = 2000
MAX_TEXT_PREVIEW = 1200

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


def init_state() -> None:
    st.session_state.setdefault("OPENAI_API_KEY", "")
    st.session_state.setdefault("DEBUG_MODE", False)
    st.session_state.setdefault("ui_location_count", 3)
    st.session_state.setdefault("debug_events", [])


def mask_secret(v: str) -> str:
    if not v:
        return ""
    if len(v) <= 8:
        return "***"
    return f"{v[:3]}***{v[-3:]}"


def debug_log(stage: str, payload: dict[str, Any]) -> None:
    if not st.session_state.get("DEBUG_MODE", False):
        return
    st.session_state["debug_events"].append({"stage": stage, "payload": payload})


def show_debug_panel() -> None:
    if not st.session_state.get("DEBUG_MODE"):
        return
    with st.expander("디버그 패널", expanded=False):
        st.caption("민감정보(API 키)는 마스킹/제외됩니다.")
        events = st.session_state.get("debug_events", [])
        if not events:
            st.write("아직 기록된 디버그 이벤트가 없습니다.")
            return
        for i, ev in enumerate(events[-60:], 1):
            st.markdown(f"**{i}. {ev['stage']}**")
            st.json(ev["payload"])


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


def _extract_output_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    return t if t else ""


def _openai_text_to_json(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1)
    return json.loads(s)


def openai_generate_plan(api_key: str, input_won: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a strict budgeting planner. Always return ONLY a valid JSON object, no markdown. "
        "Currency is KRW and period is fixed to 28 days and 4 weeks. "
        "monthly_targets sum must equal target_total."
    )
    user_prompt = {
        "task": "Generate monthly spending plan.",
        "requirements": {
            "period_days": 28,
            "period_weeks": 4,
            "currency": "KRW",
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
    text = _extract_output_text(resp)
    debug_log("openai_generate_plan_response_preview", {"text_preview": text[:MAX_TEXT_PREVIEW]})
    return _openai_text_to_json(text)


def verify_plan(plan: dict[str, Any], input_won: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    dbg: dict[str, Any] = {}
    monthly_targets = plan.get("monthly_targets", {})
    if any(k not in monthly_targets for k in CATEGORY_KEYS):
        errors.append("필수 월별 항목이 누락되었습니다.")
        return False, errors, dbg

    target_total = input_won["target_total_spending_won"]
    sum_monthly = int(sum(int(monthly_targets[k]) for k in CATEGORY_KEYS))
    dbg["sum_monthly"] = sum_monthly
    dbg["target_total"] = target_total

    if sum_monthly != target_total:
        errors.append(f"월별 항목 합계({sum_monthly:,}원)가 목표({target_total:,}원)와 다릅니다.")

    per_meal_budget = int(plan.get("eating_out", {}).get("per_meal_budget", 0))
    snack_daily = int(plan.get("daily_budgets", {}).get("snack_daily", 0))
    dbg["per_meal_budget"] = per_meal_budget
    dbg["snack_daily"] = snack_daily
    if per_meal_budget < MIN_PER_MEAL:
        errors.append(f"한 끼당 외식비는 최소 {MIN_PER_MEAL}원이어야 합니다.")
    if snack_daily < MIN_SNACK_DAILY:
        errors.append(f"하루 간식비는 최소 {MIN_SNACK_DAILY}원이어야 합니다.")

    floors = input_won["floors_won"]
    for k in ["groceries", "nightlife", "transport", "shopping", "culture", "etc", "snack", "eating_out"]:
        value = int(monthly_targets.get(k, 0))
        floor = int(floors.get(k, 0))
        dbg[f"floor_{k}"] = {"value": value, "floor": floor}
        if value < floor:
            errors.append(f"{k} 항목이 하한선을 만족하지 못했습니다. ({value:,} < {floor:,})")

    if int(monthly_targets.get("housing", 0)) < int(input_won["current_categories_won"]["housing"]):
        errors.append("거주비가 현재 거주비보다 낮게 설정되었습니다.")

    return len(errors) == 0, errors, dbg


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
    fallback = {
        "restaurant_keywords": ["맛집", "한식", "백반", "국밥", "김밥", "분식", "중식", "덮밥"],
        "cafe_keywords": ["카페", "커피", "디저트", "베이커리", "케이크", "도넛", "마카롱", "빙수"],
    }
    prompt = {
        "task": "Suggest Korean local search keywords",
        "location_label": location_label,
        "restaurant_price_band": eat_band,
        "cafe_price_band": snack_band,
        "format": {"restaurant_keywords": ["string"], "cafe_keywords": ["string"]},
        "rules": "JSON only, at least 8 each",
    }
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            temperature=0.3,
        )
        txt = _extract_output_text(resp)
        data = _openai_text_to_json(txt)
        rk = [x for x in data.get("restaurant_keywords", []) if str(x).strip()]
        ck = [x for x in data.get("cafe_keywords", []) if str(x).strip()]
        for kw in fallback["restaurant_keywords"]:
            if kw not in rk:
                rk.append(kw)
        for kw in fallback["cafe_keywords"]:
            if kw not in ck:
                ck.append(kw)
        out = {"restaurant_keywords": rk[:8], "cafe_keywords": ck[:8]}
        debug_log("openai_infer_keywords", {"location": location_label, "output": out})
        return out
    except Exception as e:
        debug_log("openai_infer_keywords_exception", {"location": location_label, "error": str(e), "tb": traceback.format_exc()[:1500]})
        return fallback


def _safe_json_preview(resp: requests.Response) -> Any:
    try:
        body = resp.json()
        txt = json.dumps(body, ensure_ascii=False)
        return txt[:MAX_TEXT_PREVIEW]
    except Exception:
        return resp.text[:MAX_TEXT_PREVIEW]


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address_kakao(address: str, kakao_key: str) -> tuple[dict[str, float] | None, dict[str, Any]]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    started = time.time()
    meta = {"url": url, "params": params, "status_code": None, "elapsed_ms": None, "response_preview": None, "error": None}
    try:
        resp = requests.get(url, headers={"Authorization": f"KakaoAK {kakao_key}"}, params=params, timeout=8)
        meta["status_code"] = resp.status_code
        meta["elapsed_ms"] = int((time.time() - started) * 1000)
        meta["response_preview"] = _safe_json_preview(resp)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if not docs:
            return None, meta
        d = docs[0]
        return {"lat": float(d["y"]), "lon": float(d["x"])}, meta
    except Exception as e:
        meta["elapsed_ms"] = int((time.time() - started) * 1000)
        meta["error"] = str(e)
        return None, meta


@st.cache_data(show_spinner=False, ttl=1800)
def search_places_kakao(
    keyword: str,
    lat: float,
    lon: float,
    kakao_key: str,
    category_group_code: str,
    size: int = 15,
    sort: str = "accuracy",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {
        "query": keyword,
        "y": lat,
        "x": lon,
        "radius": RADIUS_M,
        "sort": sort,
        "category_group_code": category_group_code,
        "size": min(size, 15),
        "page": 1,
    }
    started = time.time()
    meta = {"url": url, "params": params, "status_code": None, "elapsed_ms": None, "response_preview": None, "error": None, "documents_count": 0}
    try:
        resp = requests.get(url, headers={"Authorization": f"KakaoAK {kakao_key}"}, params=params, timeout=8)
        meta["status_code"] = resp.status_code
        meta["elapsed_ms"] = int((time.time() - started) * 1000)
        meta["response_preview"] = _safe_json_preview(resp)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        meta["documents_count"] = len(docs)
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
        return results, meta
    except Exception as e:
        meta["elapsed_ms"] = int((time.time() - started) * 1000)
        meta["error"] = str(e)
        return [], meta


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
def naver_blog_total(query: str, naver_id: str, naver_secret: str) -> tuple[int, dict[str, Any]]:
    url = "https://openapi.naver.com/v1/search/blog.json"
    params = {"query": query, "display": 1, "start": 1, "sort": "sim"}
    started = time.time()
    meta = {"url": url, "params": params, "status_code": None, "elapsed_ms": None, "response_preview": None, "error": None}
    if not naver_id or not naver_secret:
        meta["error"] = "NAVER key missing"
        return 0, meta
    try:
        resp = requests.get(
            url,
            headers={"X-Naver-Client-Id": naver_id, "X-Naver-Client-Secret": naver_secret},
            params=params,
            timeout=8,
        )
        meta["status_code"] = resp.status_code
        meta["elapsed_ms"] = int((time.time() - started) * 1000)
        meta["response_preview"] = _safe_json_preview(resp)
        resp.raise_for_status()
        return int(resp.json().get("total", 0)), meta
    except Exception as e:
        meta["elapsed_ms"] = int((time.time() - started) * 1000)
        meta["error"] = str(e)
        return 0, meta


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
    location_debug: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    blog_metas = []
    for c in candidates:
        dist = haversine_distance_m(origin["lat"], origin["lon"], c["lat"], c["lon"])
        c["distance_m"] = int(round(dist))
        c["p1"] = normalize_linear_distance_to_1_100(dist)
        total, meta = naver_blog_total(c["name"], naver_id, naver_secret)
        blog_metas.append({"place": c["name"], "meta": meta, "total": total})
        c["blog_total"] = total

    p2_list = normalize_minmax_to_1_100([c["blog_total"] for c in candidates])
    for idx, c in enumerate(candidates):
        c["p2"] = p2_list[idx]
        c["score"] = round(0.5 * c["p1"] + 0.5 * c["p2"], 2)
        c["reason"] = f"예산 구간 적합 / 거리 점수 {c['p1']} / 블로그 언급 점수 {c['p2']}"

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if location_debug is not None:
        location_debug.setdefault("naver_blog_calls", []).extend(blog_metas[:8])
    return candidates


def recommend_for_locations(
    api_key: str,
    env_keys: dict[str, str],
    plan: dict[str, Any],
    locations: list[dict[str, str]],
    exclusion_by_location: dict[str, dict[str, set[str]]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
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
        diag = {
            "geocode_ok": False,
            "candidates_raw_count": 0,
            "after_radius_filter_count": 0,
            "after_exclusion_count": 0,
            "after_scoring_count": 0,
            "final_recommended_count": 0,
            "exclusion_rest_size": len(exclusion_by_location.get(label, {}).get("restaurants", set())),
            "exclusion_cafe_size": len(exclusion_by_location.get(label, {}).get("cafes", set())),
            "api_calls": [],
        }

        origin, geo_meta = geocode_address_kakao(address, kakao_key)
        diag["api_calls"].append({"step": "geocode", "meta": geo_meta})
        if not origin:
            output[label] = {
                "restaurants": [],
                "cafes": [],
                "messages": {
                    "restaurants": "조건에 맞는 가게를 찾지 못했습니다. (주소 지오코딩 실패)",
                    "cafes": "조건에 맞는 가게를 찾지 못했습니다. (주소 지오코딩 실패)",
                },
                "diagnostics": diag,
            }
            continue
        diag["geocode_ok"] = True

        kws = openai_infer_keywords(api_key, label, eat_band, snack_band)

        def _collect(category_code: str, keywords: list[str], exclusions: set[str], fallback: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            local = {
                "raw": 0,
                "after_radius": 0,
                "after_exclusion": 0,
                "api_meta": [],
                "zero_reason": None,
            }
            all_items: list[dict[str, Any]] = []
            for kw in keywords:
                items, meta = search_places_kakao(kw, origin["lat"], origin["lon"], kakao_key, category_code, 15, "accuracy")
                local["api_meta"].append({"kw": kw, "meta": meta})
                all_items.extend(items)
            if not all_items:
                for kw in fallback:
                    items, meta = search_places_kakao(kw, origin["lat"], origin["lon"], kakao_key, category_code, 15, "accuracy")
                    local["api_meta"].append({"kw": f"fallback:{kw}", "meta": meta})
                    all_items.extend(items)

            local["raw"] = len(all_items)
            in_radius = [x for x in all_items if haversine_distance_m(origin["lat"], origin["lon"], x["lat"], x["lon"]) <= RADIUS_M]
            local["after_radius"] = len(in_radius)
            dedup: list[dict[str, Any]] = []
            seen = set()
            for x in in_radius:
                if x["id"] in seen:
                    continue
                seen.add(x["id"])
                if x["id"] in exclusions:
                    continue
                dedup.append(x)
            local["after_exclusion"] = len(dedup)

            if 0 < len(dedup) < 5:
                relaxed = ["맛집", "음식점"] if category_code == "FD6" else ["카페", "디저트"]
                for kw in relaxed:
                    items, meta = search_places_kakao(kw, origin["lat"], origin["lon"], kakao_key, category_code, 15, "distance")
                    local["api_meta"].append({"kw": f"relaxed:{kw}", "meta": meta})
                    for x in items:
                        if x["id"] not in seen and x["id"] not in exclusions:
                            if haversine_distance_m(origin["lat"], origin["lon"], x["lat"], x["lon"]) <= RADIUS_M:
                                seen.add(x["id"])
                                dedup.append(x)

            if len(dedup) == 0 and len(exclusions) > 0 and local["after_radius"] > 0:
                local["zero_reason"] = "제외 목록이 후보를 모두 제거했습니다"
            return dedup, local

        rest_ex = exclusion_by_location.get(label, {}).get("restaurants", set())
        cafe_ex = exclusion_by_location.get(label, {}).get("cafes", set())
        rest_candidates, rest_local = _collect(
            "FD6", kws["restaurant_keywords"], rest_ex, ["맛집", "한식", "백반", "국밥", "김밥", "분식", "중식", "덮밥"]
        )
        cafe_candidates, cafe_local = _collect(
            "CE7", kws["cafe_keywords"], cafe_ex, ["카페", "커피", "디저트", "베이커리", "케이크", "도넛", "마카롱", "빙수"]
        )

        diag["api_calls"].extend([
            {"step": "restaurant_search", "meta": rest_local["api_meta"][:20]},
            {"step": "cafe_search", "meta": cafe_local["api_meta"][:20]},
        ])
        diag["candidates_raw_count"] = rest_local["raw"] + cafe_local["raw"]
        diag["after_radius_filter_count"] = rest_local["after_radius"] + cafe_local["after_radius"]
        diag["after_exclusion_count"] = rest_local["after_exclusion"] + cafe_local["after_exclusion"]

        ranked_rest = rank_places(origin, rest_candidates, naver_id, naver_secret, diag)[:5]
        ranked_cafe = rank_places(origin, cafe_candidates, naver_id, naver_secret, diag)[:5]
        diag["after_scoring_count"] = len(rest_candidates) + len(cafe_candidates)
        diag["final_recommended_count"] = len(ranked_rest) + len(ranked_cafe)

        msg_rest = None
        msg_cafe = None
        if len(ranked_rest) == 0:
            msg_rest = "조건에 맞는 가게를 찾지 못했습니다."
            if rest_local.get("zero_reason"):
                msg_rest += f" ({rest_local['zero_reason']})"
        if len(ranked_cafe) == 0:
            msg_cafe = "조건에 맞는 가게를 찾지 못했습니다."
            if cafe_local.get("zero_reason"):
                msg_cafe += f" ({cafe_local['zero_reason']})"

        output[label] = {
            "restaurants": ranked_rest,
            "cafes": ranked_cafe,
            "messages": {"restaurants": msg_rest, "cafes": msg_cafe},
            "diagnostics": diag,
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

    if int(plan["checks"].get("target_total", 0)) != int(plan["checks"].get("sum_monthly_targets", 0)):
        st.warning("월 총 소비액과 항목 합계가 일치하지 않습니다. 입력 또는 생성 결과를 다시 확인해주세요.")


def main() -> None:
    init_state()
    st.set_page_config(page_title="절약 플래너", layout="wide")

    st.title("절약 플래너(임시)")
    st.write(
        "절약은 좋지만, 무조건 절약하다보면 삶의 품질이 매우 저하되기 마련입니다. 여러분의 현재 소비액과 절약 목표를 넣어 "
        "그것이 실현 가능한 목표인지, 그 목표를 이루기 위해서는 한 달 동안 어떻게 살아야 하는지 알아보세요.\n\n"
        "여러분의 수입, 소비액, 절약 목표 등을 입력하면 한 달간의 소비 플랜과 해당 플랜에 맞는 가게를 추천해드립니다."
    )
    st.caption("사이드바에서 OpenAI API key를 반드시 입력해주세요!")
    st.divider()

    env_keys = load_env_keys()

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("OPENAI_API_KEY", ""))
        st.session_state["OPENAI_API_KEY"] = api_key.strip()
        st.session_state["DEBUG_MODE"] = st.checkbox("디버그 모드", value=st.session_state.get("DEBUG_MODE", False))
        if st.session_state.get("DEBUG_MODE") and st.button("제외 목록 초기화"):
            st.session_state["excluded_ids"] = {}
            st.session_state["recommendations"] = None
            st.success("추천 제외 목록을 초기화했습니다. (플랜 유지)")

    show_debug_panel()
    input_data = _build_input_data()

    if st.button("플랜 생성"):
        st.session_state["debug_events"] = []
        if not st.session_state.get("OPENAI_API_KEY"):
            st.error("OpenAI API Key를 사이드바에 입력해주세요.")
            st.warning("사이드바에 OpenAI API key를 입력해주세요!")
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
            st.stop()

        with st.spinner("플랜 생성 중..."):
            try:
                plan = openai_generate_plan(st.session_state["OPENAI_API_KEY"], input_won)
            except json.JSONDecodeError as e:
                st.error("OpenAI 응답이 JSON 형식이 아님/파싱 실패")
                if st.session_state.get("DEBUG_MODE"):
                    st.exception(e)
                st.stop()
            except Exception as e:
                st.error(f"OpenAI 플랜 생성 실패: {e}")
                if st.session_state.get("DEBUG_MODE"):
                    st.exception(e)
                st.stop()

        verified, errors, vdbg = verify_plan(plan, input_won)
        if not verified:
            st.error("플랜 검증 실패")
            st.write("실패 항목:")
            for msg in errors:
                st.write(f"- {msg}")
            if st.session_state.get("DEBUG_MODE"):
                st.json(vdbg)
            st.stop()

        st.session_state["plan"] = plan
        st.session_state["last_input_won"] = input_won
        st.session_state["recommendations"] = None
        st.session_state["excluded_ids"] = {}

    if "plan" in st.session_state:
        _render_plan(st.session_state["plan"])

        if st.button("플랜에 맞는 가게 추천받기"):
            with st.spinner("추천 생성 중..."):
                try:
                    st.session_state["recommendations"] = recommend_for_locations(
                        api_key=st.session_state["OPENAI_API_KEY"],
                        env_keys=env_keys,
                        plan=st.session_state["plan"],
                        locations=st.session_state["last_input_won"]["locations"],
                        exclusion_by_location=st.session_state.get("excluded_ids", {}),
                    )
                except Exception as e:
                    st.error(f"추천 생성 실패: {e}")
                    if st.session_state.get("DEBUG_MODE"):
                        st.exception(e)

    if st.session_state.get("recommendations"):
        st.subheader("추천 결과")
        for label, rec in st.session_state["recommendations"].items():
            st.markdown(f"### 위치: {label}")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("식당")
                if rec.get("messages", {}).get("restaurants"):
                    st.warning(rec["messages"]["restaurants"])
                _render_place_list(rec["restaurants"])
            with c2:
                st.subheader("카페 및 디저트")
                if rec.get("messages", {}).get("cafes"):
                    st.warning(rec["messages"]["cafes"])
                _render_place_list(rec["cafes"])

            diag = rec.get("diagnostics", {})
            if st.session_state.get("DEBUG_MODE"):
                st.caption("파이프라인 단계별 카운트")
                st.table(
                    pd.DataFrame(
                        [
                            {
                                "geocode_ok": diag.get("geocode_ok"),
                                "candidates_raw_count": diag.get("candidates_raw_count"),
                                "after_radius_filter_count": diag.get("after_radius_filter_count"),
                                "after_exclusion_count": diag.get("after_exclusion_count"),
                                "after_scoring_count": diag.get("after_scoring_count"),
                                "final_recommended_count": diag.get("final_recommended_count"),
                                "exclusion_rest_size": diag.get("exclusion_rest_size"),
                                "exclusion_cafe_size": diag.get("exclusion_cafe_size"),
                            }
                        ]
                    )
                )
                st.json(diag)

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
                    if st.session_state.get("DEBUG_MODE"):
                        st.exception(e)


if __name__ == "__main__":
    main()
