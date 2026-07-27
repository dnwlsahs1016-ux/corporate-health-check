from __future__ import annotations

# (sj_div 후보들, account_id, account_nm에 포함될 fallback 문자열)
LINE_ITEMS: dict[str, dict] = {
    "assets": {"sj_div": {"BS"}, "account_id": "ifrs-full_Assets", "name_contains": "자산총계"},
    "current_assets": {
        "sj_div": {"BS"},
        "account_id": "ifrs-full_CurrentAssets",
        "name_contains": "유동자산",
    },
    "liabilities": {
        "sj_div": {"BS"},
        "account_id": "ifrs-full_Liabilities",
        "name_contains": "부채총계",
    },
    "current_liabilities": {
        "sj_div": {"BS"},
        "account_id": "ifrs-full_CurrentLiabilities",
        "name_contains": "유동부채",
    },
    "equity": {"sj_div": {"BS"}, "account_id": "ifrs-full_Equity", "name_contains": "자본총계"},
    "revenue": {
        "sj_div": {"CIS", "IS"},
        "account_id": "ifrs-full_Revenue",
        "name_contains": "매출액",
    },
    "operating_income": {
        "sj_div": {"CIS", "IS"},
        "account_id": "dart_OperatingIncomeLoss",
        "name_contains": "영업이익",
    },
    "net_income": {
        "sj_div": {"CIS", "IS"},
        "account_id": "ifrs-full_ProfitLoss",
        "name_contains": "당기순이익",
    },
    "finance_costs": {
        "sj_div": {"CIS", "IS"},
        "account_id": "ifrs-full_FinanceCosts",
        "name_contains": "금융비용",
    },
    "operating_cash_flow": {
        "sj_div": {"CF"},
        "account_id": "ifrs-full_CashFlowsFromUsedInOperatingActivities",
        "name_contains": "영업활동",
    },
    "retained_earnings": {
        "sj_div": {"BS"},
        "account_id": "ifrs-full_RetainedEarnings",
        "name_contains": "이익잉여금",
    },
}


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def extract_line_items(records: list[dict]) -> dict[str, float | None]:
    """DART fnlttSinglAcntAll 원자료에서 핵심 계정과목 금액(당기)을 뽑는다."""
    result: dict[str, float | None] = {key: None for key in LINE_ITEMS}

    for key, spec in LINE_ITEMS.items():
        match = None
        for record in records:
            if record.get("sj_div") not in spec["sj_div"]:
                continue
            if record.get("account_id") == spec["account_id"]:
                match = record
                break
        if match is None:
            for record in records:
                if record.get("sj_div") not in spec["sj_div"]:
                    continue
                if spec["name_contains"] in (record.get("account_nm") or ""):
                    match = record
                    break
        if match is not None:
            result[key] = _to_float(match.get("thstrm_amount"))

    return result


CAPITAL_IMPAIRMENT_DEBT_RATIO = 10.0  # 자본잠식 시 부채비율(부채/자기자본) 부호·크기 왜곡 방지용 상한(=1000%)
CAPITAL_IMPAIRMENT_ROE = -5.0  # 자본잠식 시 ROE 부호 왜곡(음수/음수=양수) 방지용 하한(=-500%)

# 분모가 0에 가까울 때(예: 이자비용이 거의 없는 무차입 우량기업) 비율이 비정상적으로
# 커지는 것을 막기 위한 winsorize(상하한 clip) 범위. 표본이 작을수록 이런 이상치 하나가
# 회귀/모델 전체를 왜곡할 수 있어 프로토타입에서는 특히 중요하다.
RATIO_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "roa": (-1.0, 1.0),
    "roe": (-5.0, 5.0),
    "operating_margin": (-2.0, 2.0),
    "debt_ratio": (-2.0, CAPITAL_IMPAIRMENT_DEBT_RATIO),
    "current_ratio": (0.0, 10.0),
    "equity_ratio": (-1.0, 1.0),
    "interest_coverage": (-50.0, 50.0),
    "ocf_to_assets": (-1.0, 1.0),
    "revenue_growth": (-1.0, 5.0),
}


def _clip(key: str, value: float | None) -> float | None:
    if value is None:
        return None
    lo, hi = RATIO_CLIP_BOUNDS[key]
    return max(lo, min(hi, value))


def compute_ratios(curr: dict[str, float | None], prev: dict[str, float | None] | None) -> dict:
    """핵심 재무비율 계산. 분모가 0/None이면 NaN(None) 반환.

    자기자본이 0 이하(자본잠식)인 경우 부채비율/ROE는 분모의 부호가 뒤집혀
    오히려 덜 위험해 보이는 착시가 생긴다(예: 순손실/음수자본 = 양수 ROE).
    이를 막기 위해 자본잠식 시에는 고정 상한/하한값으로 대체하고,
    별도의 capital_impairment 플래그(0/1)로 모델에 직접 노출한다.
    """

    def safe_div(a, b):
        if a is None or b in (None, 0):
            return None
        return a / b

    equity = curr["equity"]
    capital_impairment = 1 if (equity is not None and equity <= 0) else 0

    if capital_impairment:
        debt_ratio = CAPITAL_IMPAIRMENT_DEBT_RATIO
        roe = CAPITAL_IMPAIRMENT_ROE
    else:
        debt_ratio = safe_div(curr["liabilities"], equity)
        roe = safe_div(curr["net_income"], equity)

    ratios = {
        "roa": safe_div(curr["net_income"], curr["assets"]),
        "roe": roe,
        "operating_margin": safe_div(curr["operating_income"], curr["revenue"]),
        "debt_ratio": debt_ratio,
        "current_ratio": safe_div(curr["current_assets"], curr["current_liabilities"]),
        "equity_ratio": safe_div(equity, curr["assets"]),
        "interest_coverage": safe_div(curr["operating_income"], curr["finance_costs"]),
        "ocf_to_assets": safe_div(curr["operating_cash_flow"], curr["assets"]),
        "revenue_growth": None,
        "capital_impairment": capital_impairment,
    }

    if prev is not None and prev.get("revenue"):
        ratios["revenue_growth"] = safe_div(
            curr["revenue"] - prev["revenue"] if curr["revenue"] is not None else None,
            prev["revenue"],
        )

    for key in RATIO_CLIP_BOUNDS:
        ratios[key] = _clip(key, ratios[key])

    return ratios


RATIO_COLUMNS = [
    "roa",
    "roe",
    "operating_margin",
    "debt_ratio",
    "current_ratio",
    "equity_ratio",
    "interest_coverage",
    "ocf_to_assets",
    "revenue_growth",
]

# 차트/해설 생성을 위한 메타데이터: 높을수록 좋은지, 표시 형식(퍼센트/배수)
RATIO_META: dict[str, dict] = {
    "roa": {"label": "ROA (총자산순이익률)", "higher_is_better": True, "format": "percent"},
    "roe": {"label": "ROE (자기자본순이익률)", "higher_is_better": True, "format": "percent"},
    "operating_margin": {"label": "영업이익률", "higher_is_better": True, "format": "percent"},
    "debt_ratio": {"label": "부채비율", "higher_is_better": False, "format": "percent"},
    "current_ratio": {"label": "유동비율", "higher_is_better": True, "format": "percent"},
    "equity_ratio": {"label": "자기자본비율", "higher_is_better": True, "format": "percent"},
    "interest_coverage": {"label": "이자보상배율", "higher_is_better": True, "format": "multiple"},
    "ocf_to_assets": {"label": "영업현금흐름/총자산", "higher_is_better": True, "format": "percent"},
    "revenue_growth": {"label": "매출액증가율", "higher_is_better": True, "format": "percent"},
}


def compute_altman_zscore(curr: dict[str, float | None]) -> float | None:
    """Altman Z'-Score(비상장/장부가 기준 변형, Altman 1983).

    원래 Z-Score(1968)는 X4(자기자본/부채)를 '자기자본 시가총액'으로 계산하지만,
    본 파이프라인은 연도별 시가총액 이력을 수집하지 않으므로 장부가 기준 자기자본을
    쓰는 Z' 변형을 사용한다. 감사·신용평가 실무에서도 비상장기업 평가에 널리 쓰이는
    공식이라 방법론적으로 통용되는 대안이다.

    Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
    X1 = 순운전자본/총자산, X2 = 이익잉여금/총자산, X3 = 영업이익(EBIT 근사)/총자산,
    X4 = 자기자본(장부가)/부채총계, X5 = 매출액/총자산

    구간: Z' > 2.9 안전, 1.23~2.9 회색지대, Z' < 1.23 부실위험.
    """
    assets = curr.get("assets")
    if not assets:
        return None

    def safe_div(a, b):
        if a is None or b in (None, 0):
            return None
        return a / b

    current_assets = curr.get("current_assets")
    current_liabilities = curr.get("current_liabilities")
    working_capital = (
        current_assets - current_liabilities
        if current_assets is not None and current_liabilities is not None
        else None
    )

    x1 = safe_div(working_capital, assets)
    x2 = safe_div(curr.get("retained_earnings"), assets)
    x3 = safe_div(curr.get("operating_income"), assets)
    x4 = safe_div(curr.get("equity"), curr.get("liabilities"))
    x5 = safe_div(curr.get("revenue"), assets)

    components = [x1, x2, x3, x4, x5]
    if any(c is None for c in components):
        return None

    weights = [0.717, 0.847, 3.107, 0.420, 0.998]
    return sum(w * c for w, c in zip(weights, components))


def altman_zone(z_score: float | None) -> str | None:
    if z_score is None:
        return None
    if z_score > 2.9:
        return "안전지대"
    if z_score >= 1.23:
        return "회색지대"
    return "부실위험지대"
