import { useState } from "react";
import type { TimelineEntry } from "../types";

export default function AltmanCompare({ entry }: { entry: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  if (entry.altman_zscore == null) return null;

  return (
    <div
      style={{
        marginTop: 12,
        padding: "10px 14px",
        borderRadius: 8,
        background: "var(--color-bg)",
        border: "1px solid var(--color-border)",
        fontSize: 13,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span style={{ color: "var(--color-text-muted)" }}>Altman Z'-Score(전통적 부실예측 공식) </span>
          <strong>{entry.altman_zscore.toFixed(2)}</strong>
          <span style={{ color: "var(--color-text-muted)" }}> ({entry.altman_zone})</span>
          {entry.altman_risk_score != null && (
            <>
              {" · "}
              <span style={{ color: "var(--color-text-muted)" }}>환산 위험점수 </span>
              <strong>{entry.altman_risk_score.toFixed(1)}</strong>
              <span style={{ color: "var(--color-text-muted)" }}>/100</span>
            </>
          )}
        </div>
        <button
          onClick={() => setOpen(!open)}
          style={{
            border: "none",
            background: "none",
            color: "var(--color-primary)",
            fontSize: 12,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          {open ? "설명 닫기 ▲" : "Altman Score란? ▼"}
        </button>
      </div>
      {open && (
        <div style={{ margin: "8px 0 0", color: "var(--color-text-muted)", lineHeight: 1.6 }}>
          <p style={{ margin: 0 }}>
            Altman Z-Score는 1968년 에드워드 알트만 교수가 만든, 지금도 감사·신용평가 실무에서
            널리 쓰이는 전통적 부실예측 공식입니다. 순운전자본·이익잉여금·영업이익·자기자본·매출액을
            총자산(또는 부채) 대비 비율로 조합해 하나의 점수(Z)로 계산하며, 이 점수가 낮을수록
            부실 위험이 크다고 봅니다(2.9 이상 안전지대, 1.23~2.9 회색지대, 1.23 미만 부실위험지대).
            이 서비스는 연도별 시가총액 이력이 없어 시장가치 대신 장부가 자기자본을 쓰는 변형인
            Z'-Score를 사용했습니다.
          </p>
          <p style={{ margin: "8px 0 0" }}>
            이 공식은 원래 제조업 상장사 데이터로 만들어져 자산 구성이 크게 다른 금융업·서비스업
            등에는 그대로 적용하기 어렵습니다. 그래서 한국표준산업분류상 제조업(코드 10~34)에
            해당하는 기업에만 Z-Score를 계산해 보여주며, 그 외 업종은 이 카드 자체가 표시되지
            않습니다.
          </p>
          <p style={{ margin: "8px 0 0" }}>
            위 "환산 위험점수"는 Z-Score 하나만 입력으로 쓰는 별도의 1변수 로지스틱회귀를 전체
            기간 데이터로 학습해서(Platt scaling과 같은 방식) 만든 값입니다 — 실제 상장폐지
            결과를 학습해 "이 Z-Score를 가진 기업이 내년 상장폐지될 확률"로 다시 계산하고, 그
            확률에 100을 곱해 우리 모델과 같은 0~100 척도로 비교할 수 있게 했습니다.
          </p>
        </div>
      )}
    </div>
  );
}
