"""기업건강검진 Streamlit 데모.

FastAPI+React 버전과 동일한 사전 계산 결과(data/processed/panel.parquet, model.joblib)를
그대로 읽어서 보여주는 단일 파일 앱. 별도 백엔드 서버 없이 Streamlit Community Cloud에
바로 배포할 수 있도록 만들었다.

실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.config import PROCESSED_DIR
from app.pipeline.benchmark import build_commentary, peer_average
from app.pipeline.features import RATIO_COLUMNS, RATIO_META, altman_zone
from app.pipeline.model import MODEL_FEATURES, explain_score, load_model, prepare_model_frame

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
MARKET_CAP_PATH = PROCESSED_DIR / "market_cap.csv"
METRICS_PATH = PROCESSED_DIR / "pipeline_metrics.json"

st.set_page_config(page_title="기업건강검진", page_icon="🩺", layout="wide")


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)


@st.cache_data
def load_market_cap() -> pd.DataFrame:
    if not MARKET_CAP_PATH.exists():
        return pd.DataFrame(columns=["corp_name", "market_cap_rank", "market_cap"])
    return pd.read_csv(MARKET_CAP_PATH)


def build_company_list(panel: pd.DataFrame) -> pd.DataFrame:
    """기준연도 내림차순 -> 시총 순위 오름차순(없으면 맨 뒤) -> 위험점수 내림차순으로 정렬한 목록."""
    latest = panel.sort_values("year").groupby("corp_name").tail(1).copy()
    market_cap = load_market_cap()
    latest = latest.merge(market_cap[["corp_name", "market_cap_rank"]], on="corp_name", how="left")
    latest["market_cap_rank"] = latest["market_cap_rank"].fillna(10**9)
    return latest.sort_values(
        ["year", "market_cap_rank", "risk_score"], ascending=[False, True, False]
    )


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

    top = contributions[0]
    direction = "위험도를 높이는" if top["contribution"] > 0 else "위험도를 낮추는"
    summary = f"**가장 큰 영향을 미친 지표는 '{top['label']}'**이며, {direction} 방향으로 작용했습니다."
    if len(contributions) > 1:
        second = contributions[1]
        direction2 = "높이는" if second["contribution"] > 0 else "낮추는"
        summary += f" 그다음으로는 '{second['label']}'이 위험도를 {direction2} 방향으로 크게 작용했습니다."
    st.markdown(summary)

    st.caption(
        "※ 학습 표본이 작아(상장폐지 확정 사례 기준) 일부 지표의 기여 방향이 재무이론과 다르게 "
        "나타날 수 있습니다. 완전자본잠식처럼 규정상 명확한 위험 요인은 규칙으로 별도 보정했습니다."
    )


def render_ratio_chart(timeline: pd.DataFrame, ratio_key: str, commentary: str):
    meta = RATIO_META[ratio_key]
    st.markdown(f"**{meta['label']}**")

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
    # 제목은 위 st.markdown으로 따로 빼서 plotly 자체 title과 상단 범례가 겹치지 않게 한다.
    fig.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
    )
    fig.update_xaxes(dtick=1, tickformat="d")  # 연도는 정수로만(2021.5 같은 소수 눈금 방지)
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

    z = latest_row.get("altman_zscore")
    if pd.notna(z):
        az_score = latest_row.get("altman_risk_score")
        line = f"Altman Z'-Score(전통적 부실예측 공식) **{z:.2f}** ({altman_zone(z)})"
        if pd.notna(az_score):
            line += f" · 환산 위험점수 **{az_score:.1f}**/100"
        st.caption(line)
        with st.expander("Altman Score란?", expanded=False):
            st.caption(
                "Altman Z-Score는 1968년 에드워드 알트만 교수가 만든, 지금도 감사·신용평가 실무에서 "
                "널리 쓰이는 전통적 부실예측 공식입니다. 순운전자본·이익잉여금·영업이익·자기자본·매출액을 "
                "총자산(또는 부채) 대비 비율로 조합해 하나의 점수(Z)로 계산하며, 점수가 낮을수록 부실 "
                "위험이 크다고 봅니다(2.9 이상 안전지대, 1.23~2.9 회색지대, 1.23 미만 부실위험지대). "
                "이 서비스는 연도별 시가총액 이력이 없어 시장가치 대신 장부가 자기자본을 쓰는 변형인 "
                "Z'-Score를 사용했습니다. '환산 위험점수'는 Z-Score를 우리 모델과 같은 0~100 척도로 "
                "비교할 수 있도록 별도의 1변수 로지스틱회귀로 보정한 값입니다."
            )

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
    st.caption(
        "각 그래프 아래 문구는 최신 연도 기준으로 동종업계·유사규모(총자산 1/3~3배, 부족하면 "
        "동종업계 전체 → 비교기업 전체 순으로 범위를 넓힘) 평균과 비교한 해설입니다."
    )

    cols = st.columns(2)
    for i, ratio_key in enumerate(RATIO_COLUMNS):
        peer_info = peer_average(panel, corp_name, latest_year, ratio_key)
        commentary = build_commentary(ratio_key, latest_row[ratio_key], peer_info)
        with cols[i % 2]:
            render_ratio_chart(company_rows, ratio_key, commentary)


def render_model_validation():
    if not METRICS_PATH.exists():
        return
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    holdout = metrics.get("temporal_holdout")
    if not holdout or "error" in holdout:
        return

    with st.expander("📊 모델 검증 결과 (시계열 학습/검증 분리)", expanded=False):
        st.caption(
            f"{holdout['split_year']}년까지의 데이터로만 학습한 뒤, {holdout['split_year'] + 1}년 이후"
            "(실제 상장폐지 사례 포함)로 검증했습니다. 학습 시점에 미래 정보를 전혀 사용하지 않았을 때도 "
            "부실기업을 구분할 수 있는지 확인하기 위한 것으로, 실제 서비스에 쓰이는 모델(전체 기간 학습)과는 "
            "별개의 평가용 모델입니다."
        )
        shown = holdout.get("threshold_tuned") or holdout.get("threshold_default")

        roc_auc_label = f"{holdout['roc_auc']:.3f}"
        ci = holdout.get("roc_auc_ci95")
        if ci:
            roc_auc_label += f" (95% CI {ci['lower']:.2f}~{ci['upper']:.2f})"

        cols = st.columns(4)
        cols[0].metric("ROC-AUC", roc_auc_label)
        if shown:
            cols[1].metric("Recall(재현율)", f"{shown['recall'] * 100:.1f}%")
            cols[2].metric("Precision(정밀도)", f"{shown['precision'] * 100:.1f}%")
        cols[3].metric(
            "검증 구간 폐지 사례",
            f"{holdout['n_test_positive']}건",
            help=f"학습 {holdout['n_train']}건(양성 {holdout['n_train_positive']}) / "
            f"검증 {holdout['n_test']}건(양성 {holdout['n_test_positive']})",
        )
        if holdout.get("pr_auc") is not None:
            st.caption(
                f"PR-AUC(양성이 희귀할 때 ROC-AUC보다 더 엄격한 지표): {holdout['pr_auc']:.3f} "
                f"(무작위 분류기 기준선 {holdout['pr_auc_baseline'] * 100:.1f}%)"
            )
        if shown:
            cm = shown["confusion_matrix"]
            st.caption(
                f"혼동행렬(임계값 {shown['threshold']:.2f} 기준): 실제 폐지 {holdout['n_test_positive']}건 중 "
                f"{cm['true_positive']}건 적중(재현율 {shown['recall']*100:.1f}%), "
                f"{cm['false_negative']}건 놓침. 위험 예측 {cm['true_positive'] + cm['false_positive']}건 중 "
                f"실제로 맞은 건 {cm['true_positive']}건(정밀도 {shown['precision']*100:.1f}%)."
            )
        default = holdout.get("threshold_default")
        tuned = holdout.get("threshold_tuned")
        if default and tuned:
            st.caption(
                "※ class_weight='balanced'를 쓴 로지스틱회귀에서는 0.5가 \"실제 위험확률 50%\"를 "
                "뜻하지 않습니다. 학습구간 내부 교차검증만으로 고른 임계값"
                f"({tuned['threshold']:.2f})을 대신 적용하면, 위험 예측 건수(오탐 포함)가 "
                f"{default['confusion_matrix']['true_positive'] + default['confusion_matrix']['false_positive']}건 → "
                f"{tuned['confusion_matrix']['true_positive'] + tuned['confusion_matrix']['false_positive']}건으로 "
                "달라집니다(재현율을 더 중시하는 F2 기준으로 골랐습니다)."
            )

        altman = metrics.get("altman_benchmark")
        if altman and "error" not in altman and altman.get("roc_auc") is not None:
            st.divider()
            st.markdown("**전통적 Altman Z'-Score 대비**")
            st.caption(
                f"같은 검증 구간({holdout['split_year'] + 1}년 이후)에서, 감사·신용평가 실무에서 "
                "흔히 쓰이는 전통적 부실예측 공식인 Altman Z'-Score를 위험 순위로 사용했을 때의 "
                "판별력과 비교했습니다."
            )
            zcols = st.columns(2)
            zcols[0].metric("로지스틱회귀 모델 AUC", f"{holdout['roc_auc']:.3f}")
            zcols[1].metric("Altman Z'-Score AUC", f"{altman['roc_auc']:.3f}")
            st.caption(
                "고정된 계수식인 Z-Score와 달리, 로지스틱 회귀는 거시조정된 지표로 데이터에 맞춰 "
                "계수를 학습해 이 검증 구간에서 더 높은 판별력을 보였습니다."
            )

        ensemble = metrics.get("ensemble_benchmark")
        if ensemble and "error" not in ensemble and ensemble.get("ensemble_auc") is not None:
            st.divider()
            st.markdown("**로지스틱회귀 + Altman Z-Score 앙상블 실험**")
            st.caption("두 모델의 예측 확률을 단순평균해서 결합하면 더 나아지는지도 같은 검증 구간에서 테스트했습니다.")
            ecols = st.columns(3)
            ecols[0].metric("로지스틱 단독", f"{ensemble['logistic_only_auc']:.3f}")
            ecols[1].metric("Altman 단독", f"{ensemble['altman_only_auc']:.3f}")
            ecols[2].metric("단순평균 앙상블", f"{ensemble['ensemble_auc']:.3f}")
            st.caption(
                "※ 정직하게 보고하면, 이 앙상블은 로지스틱 단독보다 오히려 낮은 AUC가 나왔습니다. "
                "Altman 단독 성능이 상대적으로 약해서 평균을 내면 로지스틱의 신호를 끌어내리는 효과가 "
                "난 것으로 보입니다. 그래서 서비스에는 앙상블 대신 로지스틱회귀 단독 점수를 사용합니다."
            )

        random_split = metrics.get("random_split_benchmark")
        if random_split and "error" not in random_split and random_split.get("roc_auc") is not None:
            st.divider()
            train_pct = round((1 - random_split["test_size"]) * 100)
            test_pct = round(random_split["test_size"] * 100)
            st.markdown(f"**(참고) 시간 순서를 무시한 무작위 {train_pct}:{test_pct} 분할 결과**")
            st.caption("기업-연도 로우를 연도 상관없이 라벨 비율을 유지한 채 무작위로 나눠 평가하면 어떻게 달라지는지도 확인했습니다.")
            rcols = st.columns(2)
            rcols[0].metric("무작위 분할 AUC", f"{random_split['roc_auc']:.3f}")
            rcols[1].metric("시계열 분할 AUC(위 기준)", f"{holdout['roc_auc']:.3f}")
            st.caption(
                "※ 무작위 분할이 시계열 분할보다 수치가 더 높게 나옵니다. 무작위 분할은 2024년 데이터가 "
                "학습에, 2020년 데이터가 검증에 들어가는 식으로 미래 정보가 은연중에 섞일 수 있어 실제보다 "
                "낙관적인 성능으로 보이기 쉽습니다. \"내년도 위험을 예측한다\"는 이 서비스의 실제 사용 "
                "시나리오와는 시계열 분할이 더 정직한 검증이라고 판단해, 대표 검증 지표로는 시계열 분할(위) "
                "결과를 사용합니다."
            )


def main():
    st.markdown(
        "<span style='font-size:30px; font-weight:800;'>기업건강검진</span> "
        "<span style='color:#DB4E18; font-weight:600;'>AI 기반 코스닥 재무위험 진단</span>",
        unsafe_allow_html=True,
    )
    st.write(
        "class-weighted 로지스틱 회귀 AI 모델이 **DART 전자공시 재무제표(2015~2025년, 코스닥 상장사 600여 곳)**, "
        "**한국은행 ECOS 기준금리**, **KRX 상장폐지 이력(2015년 이후 재무적 사유로 폐지된 100여 건)** 데이터를 "
        "학습했습니다. 거시경제(기준금리) 요인을 제거한 기업 고유의 재무위험을 계산해, 내년도 상장폐지 "
        "위험점수를 보여줍니다."
    )
    st.warning(
        "⚠️ 소규모 표본으로 학습된 실험적 모델의 결과이며 통계적으로 검증되지 않았습니다. "
        "**실제 투자 판단, 신용평가, 거래 의사결정에 사용하지 마세요.** "
        "표시된 기업명은 모델 검증을 위한 예시일 뿐, 해당 기업의 재무 건전성을 공식적으로 나타내지 않습니다."
    )

    if not PANEL_PATH.exists():
        st.error("파이프라인 결과가 없습니다. scripts/run_pipeline.py를 먼저 실행하세요.")
        return

    panel = load_panel()
    bundle = get_model_bundle()

    if "selected_corp" not in st.session_state:
        st.session_state.selected_corp = None

    if st.session_state.selected_corp is not None:
        if st.button("← 목록으로"):
            st.session_state.selected_corp = None
            st.rerun()
        st.divider()
        render_company_detail(panel, st.session_state.selected_corp, bundle)
        return

    render_model_validation()

    latest = build_company_list(panel)
    latest["risk_label"] = latest["risk_score"].apply(risk_label)

    search = st.text_input("기업명 검색", "")

    filter_cols = st.columns(2)
    with filter_cols[0]:
        category_filter = st.multiselect(
            "구분 필터",
            options=["상장폐지 사례(검증용)", "건전기업 벤치마크"],
            default=["상장폐지 사례(검증용)", "건전기업 벤치마크"],
        )
    with filter_cols[1]:
        risk_filter = st.multiselect(
            "위험도 필터", options=["고위험", "주의", "안전"], default=["고위험", "주의", "안전"]
        )

    category_map = {
        "상장폐지 사례(검증용)": "financial_distress",
        "건전기업 벤치마크": "healthy_benchmark",
    }

    table = latest.copy()
    if search:
        table = table[table["corp_name"].str.contains(search)]
    if category_filter:
        table = table[table["category"].isin([category_map[c] for c in category_filter])]
    else:
        table = table.iloc[0:0]
    if risk_filter:
        table = table[table["risk_label"].isin(risk_filter)]
    else:
        table = table.iloc[0:0]

    st.caption(f"{len(table)}개 기업 표시 중 · 표에서 기업 행을 클릭하면 상세 위험도 분석을 볼 수 있습니다.")

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
        height=520,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection and event.selection.get("rows"):
        st.session_state.selected_corp = table.iloc[event.selection["rows"][0]]["corp_name"]
        st.rerun()


if __name__ == "__main__":
    main()
