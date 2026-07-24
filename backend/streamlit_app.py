"""기업건강검진 Streamlit 데모.

FastAPI+React 버전과 동일한 사전 계산 결과(data/processed/panel.parquet, model.joblib)를
그대로 읽어서 보여주는 단일 파일 앱. 별도 백엔드 서버 없이 Streamlit Community Cloud에
바로 배포할 수 있도록 만들었다.

실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.config import PROCESSED_DIR
from app.pipeline.benchmark import build_commentary, peer_average
from app.pipeline.features import RATIO_COLUMNS, RATIO_META
from app.pipeline.model import MODEL_FEATURES, explain_score, load_model, prepare_model_frame

PANEL_PATH = PROCESSED_DIR / "panel.parquet"

st.set_page_config(page_title="기업건강검진", page_icon="🩺", layout="wide")


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)


@st.cache_resource
def get_model_bundle():
    return load_model()


def risk_color(score: float) -> str:
    if score >= 60:
        return "#C0392B"
    if score >= 30:
        return "#D9930F"
    return "#1E8A5F"


def risk_label(score: float) -> str:
    if score >= 60:
        return "고위험"
    if score >= 30:
        return "주의"
    return "안전"


def render_gauge(score: float):
    color = risk_color(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#eaf6f0"},
                    {"range": [30, 60], "color": "#fdf3e0"},
                    {"range": [60, 100], "color": "#fbeae7"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<p style='text-align:center; color:{color}; font-weight:700; margin-top:-16px;'>"
        f"{risk_label(score)}</p>",
        unsafe_allow_html=True,
    )


def render_risk_explanation(explanation: dict):
    st.subheader("위험도 산출 근거")
    st.caption(
        f"모델 예측 확률 {explanation['model_probability'] * 100:.1f}% "
        f"(모델 점수 {explanation['model_score']}점)"
        + (
            f" → 규칙 조정 후 최종 {explanation['final_risk_score']}점"
            if explanation.get("rule_adjusted")
            else ""
        )
        + ". 각 재무지표(거시조정 고유위험)가 위험도를 얼마나 끌어올리거나 낮췄는지 보여줍니다."
    )

    if explanation.get("rule_note"):
        st.warning(f"⚠ {explanation['rule_note']}")

    contributions = explanation["contributions"]
    labels = [c["label"] for c in contributions][::-1]
    values = [c["contribution"] for c in contributions][::-1]
    colors = [risk_color(60) if v > 0 else risk_color(0) for v in values]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="위험도(로그오즈) 기여도",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "※ 학습 표본이 작아(상장폐지 확정 사례 기준) 일부 지표의 기여 방향이 재무이론과 다르게 "
        "나타날 수 있습니다. 완전자본잠식처럼 규정상 명확한 위험 요인은 규칙으로 별도 보정했습니다."
    )


def render_ratio_chart(timeline: pd.DataFrame, ratio_key: str, commentary: str):
    meta = RATIO_META[ratio_key]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["year"], y=timeline[ratio_key], mode="lines", name="원지표",
            line=dict(color="#9a948d", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeline["year"], y=timeline[f"{ratio_key}_idio"], mode="lines", name="고유위험",
            line=dict(color="#DB4E18", width=2),
        )
    )
    fig.update_layout(
        title=meta["label"], height=260, margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(commentary)


def render_company_detail(panel: pd.DataFrame, corp_name: str, bundle: dict):
    company_rows = panel[panel["corp_name"] == corp_name].sort_values("year")
    latest_row = company_rows.iloc[-1]
    latest_year = int(latest_row["year"])

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        title = corp_name
        if latest_row["capital_impairment"]:
            title += "  :red[⚠ 완전자본잠식]"
        st.markdown(f"## {title}")
        meta_line = []
        if pd.notna(latest_row.get("stock_code")):
            meta_line.append(f"종목코드 {latest_row['stock_code']}")
        if pd.notna(latest_row.get("delisting_date")) and latest_row.get("delisting_date"):
            meta_line.append(f"상장폐지일 {latest_row['delisting_date']}")
        st.caption(" · ".join(meta_line) if meta_line else "")
    with header_cols[1]:
        render_gauge(float(latest_row["risk_score"]))

    prepared_panel = prepare_model_frame(panel)
    prepared_row = prepared_panel[
        (prepared_panel["corp_name"] == corp_name) & (prepared_panel["year"] == latest_year)
    ].iloc[-1]
    feature_values = prepared_row[MODEL_FEATURES].to_dict()
    explanation = explain_score(bundle, feature_values)
    explanation["rule_adjusted"] = bool(latest_row.get("rule_adjusted", False))
    explanation["final_risk_score"] = latest_row["risk_score"]
    if explanation["rule_adjusted"]:
        explanation["rule_note"] = (
            "완전자본잠식 상태로 판단되어, 모델 예측 점수와 무관하게 위험점수 하한선(70점)을 적용했습니다."
        )
    render_risk_explanation(explanation)

    st.subheader("재무지표 추이 (원지표 vs 거시조정 고유위험)")
    st.caption("각 그래프 아래 문구는 최신 연도 기준으로 동종업계(부족하면 비교기업 전체) 평균과 비교한 해설입니다.")

    cols = st.columns(2)
    for i, ratio_key in enumerate(RATIO_COLUMNS):
        peer_info = peer_average(panel, corp_name, latest_year, ratio_key)
        commentary = build_commentary(ratio_key, latest_row[ratio_key], peer_info)
        with cols[i % 2]:
            render_ratio_chart(company_rows, ratio_key, commentary)


def main():
    st.markdown(
        "<span style='font-size:30px; font-weight:800;'>기업건강검진</span> "
        "<span style='color:#DB4E18; font-weight:600;'>코스닥 재무위험 진단</span>",
        unsafe_allow_html=True,
    )
    st.write(
        "한국은행 ECOS·DART 재무제표 데이터를 기반으로 거시경제 요인을 제거한 기업 고유의 재무위험을 "
        "계산해, 내년도 상장폐지 위험점수를 보여줍니다."
    )

    if not PANEL_PATH.exists():
        st.error("파이프라인 결과가 없습니다. scripts/run_pipeline.py를 먼저 실행하세요.")
        return

    panel = load_panel()
    bundle = get_model_bundle()

    latest = panel.sort_values("year").groupby("corp_name").tail(1).sort_values(
        "risk_score", ascending=False
    )

    if "selected_corp" not in st.session_state:
        st.session_state.selected_corp = latest.iloc[0]["corp_name"]

    search = st.text_input("기업명 검색", "")
    table = latest[latest["corp_name"].str.contains(search)] if search else latest

    display_df = table[["corp_name", "stock_code", "category", "year", "risk_score", "capital_impairment"]].rename(
        columns={
            "corp_name": "기업명",
            "stock_code": "종목코드",
            "category": "구분",
            "year": "기준연도",
            "risk_score": "위험점수",
            "capital_impairment": "완전자본잠식",
        }
    )
    display_df["구분"] = display_df["구분"].map(
        {"financial_distress": "상장폐지 사례(검증용)", "healthy_benchmark": "건전기업 벤치마크"}
    )

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=380,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection and event.selection.get("rows"):
        st.session_state.selected_corp = table.iloc[event.selection["rows"][0]]["corp_name"]

    st.divider()
    render_company_detail(panel, st.session_state.selected_corp, bundle)


if __name__ == "__main__":
    main()
