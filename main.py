# main.py
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------
# 한글 폰트 (Streamlit CSS)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 데이터 로딩 (인코딩 이중 시도 + 캐싱)
@st.cache_data
def read_csv_cached(file):
    """바이트를 꺼내 utf-8-sig → cp949 순서로 시도."""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def safe_read(file):
    """업로드된 파일이 있으면 읽고, 없으면 빈 DataFrame 반환."""
    if file is None:
        return pd.DataFrame()
    with st.spinner("데이터를 읽어오는 중…"):
        df = read_csv_cached(file)
    return df


# ----------------------------------------------------------------------
# 열 이름 표준화 (KeyError 방지) — 문서 예시 코드 그대로
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed

    return df


# ----------------------------------------------------------------------
# 학교명 추출 (파일명 앞부분에서)
def extract_school_name(filename):
    if not filename:
        return None
    base = filename.split(".")[0]
    parts = re.split(r"[_.\-]+", base)
    for p in parts:
        p = p.strip()
        if p and not p[0].isdigit():
            return p
    return parts[0].strip() if parts else None


# ----------------------------------------------------------------------
# 그래프 테마 (Plotly 한글 폰트)
def set_fig_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig


# ----------------------------------------------------------------------
# 도우미: 있는 열만 골라서 리스트로 반환
def present_cols(df, candidates):
    return [c for c in candidates if c in df.columns]


# ----------------------------------------------------------------------
# 도우미: 숫자 데이터만 뽑아 NaN 제거
def clean_nums(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s


# ----------------------------------------------------------------------
# 상관계수 + 추세선 계산
def correlation_with_trend(x, y):
    xf = clean_nums(x)
    yf = clean_nums(y)
    if len(xf) < 2 or len(yf) < 2:
        return None, None, None, None, None
    df_t = pd.DataFrame({
        "x": xf.reset_index(drop=True),
        "y": yf.reset_index(drop=True),
    })
    df_t = df_t.dropna()
    if len(df_t) < 2:
        return None, None, None, None, None
    r = df_t["x"].corr(df_t["y"])
    slope, intercept = np.polyfit(df_t["x"], df_t["y"], 1)
    x_line = np.array([df_t["x"].min(), df_t["x"].max()])
    y_line = slope * x_line + intercept
    return r, slope, intercept, x_line, y_line


# ----------------------------------------------------------------------
# 산점도 + 추세선 (재사용 함수)
def scatter_with_trend(df, x_col, y_col, x_label, y_label):
    r, slope, intercept, x_line, y_line = correlation_with_trend(
        df[x_col], df[y_col]
    )
    if r is None:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="데이터가 부족하여 산점도를 표시할 수 없어요",
            showarrow=False,
            font=dict(size=14),
        )
        set_fig_font(fig)
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=400,
        )
        return fig, None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col].dropna(),
            y=df[y_col].dropna(),
            mode="markers",
            marker=dict(size=8, opacity=0.7),
            name="개체",
            hoverinfo="x+y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="red", width=2),
            name=f"추세선 (r = {r:.3f})",
        )
    )
    set_fig_font(fig)
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig, r


# ----------------------------------------------------------------------
# 히스토그램 + 통계 요약 (Tab 1용)
def histogram_with_stats(df, col, title, unit):
    s = clean_nums(df[col])
    if len(s) < 2:
        return None, None

    mean_val = float(s.mean())
    std_val = float(s.std())
    var_val = float(s.var())
    min_val = float(s.min())
    max_val = float(s.max())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=s.values,
            nbinsx=max(10, min(30, len(s) // 3)),
            marker_color="skyblue",
            opacity=0.8,
            name="빈도",
        )
    )
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"평균 {mean_val:.2f}{unit}",
        annotation_position="top",
    )
    set_fig_font(fig)
    fig.update_layout(
        title=f"{title} 분포",
        xaxis_title=f"{title} ({unit})",
        yaxis_title="개체 수",
        height=360,
        showlegend=False,
    )
    return fig, {
        "평균": mean_val,
        "표준편차": std_val,
        "분산": var_val,
        "최소값": min_val,
        "최대값": max_val,
        "평균과의 차이(최소)": mean_val - min_val,
        "평균과의 차이(최대)": max_val - mean_val,
    }


# ----------------------------------------------------------------------
# CSV 다운로드 버튼 (고유한 key 사용 → 중복 ID 방지)
def download_button(df, label, file_name, key):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


# ----------------------------------------------------------------------
# 메인 앱
def main():
    st.set_page_config(page_title="송도고 극지식물 생육 분석", layout="wide")
    st.title("🌱 송도고등학교 그린 3팀 2026 — 극지식물 생육 차이 분석")

    # 사이드바: CSV 업로드
    st.sidebar.header("📁 데이터 업로드")
    env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 CSV 파일을 올리면 자동으로 분석돼요.\n"
        "인코딩(utf-8-sig / cp949)과 열 이름은 자동으로 처리됩니다."
    )

    # 데이터 미리 읽기 (한 번만)
    env_raw = safe_read(env_file)
    growth_raw = safe_read(growth_file)

    env_df = std_cols(env_raw) if not env_raw.empty else pd.DataFrame()
    growth_df = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

    school_name = extract_school_name(
        env_file.name if env_file else (growth_file.name if growth_file else None)
    )
    if school_name:
        st.caption(f"🏫 분석 대상: {school_name}")

    tabs = st.tabs([
        "📊 Tab 1 — 생중량 차이는 얼마나 날까?",
        "📏 Tab 2 — 지상부 길이 vs 잎 수",
        "🍃 Tab 3 — 지하부 무게 vs 잎 수",
    ])

    # ==============================================================
    # Tab 1: 생중량 분포 + 통계
    # ==============================================================
    with tabs[0]:
        try:
            st.header("📊 개체들마다 생중량 차이는 얼마나 날까?")

            st.markdown("""
**생중량**이란 식물을 바로 쟀을 때의 무게예요 (수분이 포함된 상태).  
개체마다 생중량이 얼마나 다른지 히스토그램으로 분포를 보고, 평균·표준편차·분산을 함께 확인해요.

- **평균**: 모든 개체 값을 더해 개수로 나눈 값 (대표값)
- **표준편차**: 평균에서 각 개체가 얼마나 떨어져 있는지 나타내는 값. 클수록 개체 간 차이가 커요
- **분산**: 표준편차를 제곱한 값 (통계 계산의 기본 단위)
            """.strip()

            )

            if growth_df.empty:
                st.info("생육 결과 CSV를 사이드바에서 올려주세요.")
            elif "생중량" not in growth_df.columns:
                st.info("생중량 열이 없어서 이 탭을 표시할 수 없어요.")
            else:
                fig, stats = histogram_with_stats(
                    growth_df,
                    "생중량",
                    "생중량",
                    "g",
                )
                if fig is None:
                    st.info("생중량 데이터가 부족해요.")
                else:
                    col_a, col_b = st.columns([1.2, 1.0])
                    with col_a:
                        st.plotly_chart(fig, width="stretch")
                    with col_b:
                        st.markdown("### 📋 통계 요약")
                        stat_rows = []
                        for k, v in stats.items():
                            stat_rows.append({"지표": k, "값": round(v, 3)})
                        st.table(pd.DataFrame(stat_rows))

                    gw = clean_nums(growth_df["생중량"])
                    if len(gw) >= 2:
                        idx_max = gw.idxmax()
                        idx_min = gw.idxmin()
                        max_label = (
                            str(growth_df.loc[idx_max, "개체번호"])
                            if "개체번호" in growth_df.columns
                            else "?"
                        )
                        min_label = (
                            str(growth_df.loc[idx_min, "개체번호"])
                            if "개체번호" in growth_df.columns
                            else "?"
                        )
                        st.markdown(f"""
**생중량 최대 개체**: {max_label} ({gw.max():.2f}g) — 평균보다 {stats['평균과의 차이(최대)']:.2f}g 더 무거워요  
**생중량 최소 개체**: {min_label} ({gw.min():.2f}g) — 평균보다 {stats['평균과의 차이(최소)']:.2f}g 더 가벼워요
                        """.strip())

                        cv = (stats["표준편차"] / stats["평균"]) * 100 if stats["평균"] != 0 else 0
                        st.markdown(f"**변동계수(CV)**: {cv:.1f}%  (표준편차를 평균으로 나눈 백분율 — 값이 클수록 개체 간 차이가 상대적으로 커요)")

                    mean_val = stats["평균"]
                    std_val = stats["표준편차"]
                    min_val = stats["최소값"]
                    max_val = stats["최대값"]
                    diff_range = max_val - min_val
                    st.markdown(f"""
---
**📝 결과 해석**

- 전체 {len(gw)}개체의 평균 생중량은 **{mean_val:.2f}g**이고, 표준편차는 **{std_val:.2f}g**예요.
- 생중량 범위는 **{min_val:.2f}g ~ {max_val:.2f}g**로, 최대와 최소의 차이는 **{diff_range:.2f}g**예요.
- 표준편차가 평균의 {(std_val / mean_val * 100):.1f}% 수준이어서, 개체 간 생육 차이가 상당히 큰 편이에요.
- 특히 최대 개체와 최소 개체의 차이가 커서, 같은 조건에서 키워도 개체마다 생육 속도에 차이가 있음을 보여줘요.
                    """.strip())

                    q75 = gw.quantile(0.75)
                    q25 = gw.quantile(0.25)
                    top_df = growth_df.loc[gw >= q75].copy()
                    low_df = growth_df.loc[gw <= q25].copy()

                    st.markdown("### 🔬 상위 25% vs 하위 25% 개체 비교")
                    comp_a, comp_b = st.columns(2)
                    with comp_a:
                        st.markdown(f"**상위 25% (≥ Q75 = {q75:.2f}g)**")
                        leaf_col = present_cols(top_df, ["잎 수"])
                        if leaf_col and len(top_df) >= 1:
                            st.metric("개체 수", f"{len(top_df)}개체")
                            st.metric(
                                "평균 잎 수",
                                f"{clean_nums(top_df[leaf_col[0]]).mean():.1f}장",
                            )
                        length_col = present_cols(top_df, ["지상부 길이"])
                        if length_col and len(top_df) >= 1:
                            st.metric(
                                "평균 지상부 길이",
                                f"{clean_nums(top_df[length_col[0]]).mean():.1f}mm",
                            )
                    with comp_b:
                        st.markdown(f"**하위 25% (≤ Q25 = {q25:.2f}g)**")
                        leaf_col_l = present_cols(low_df, ["잎 수"])
                        if leaf_col_l and len(low_df) >= 1:
                            st.metric("개체 수", f"{len(low_df)}개체")
                            st.metric(
                                "평균 잎 수",
                                f"{clean_nums(low_df[leaf_col_l[0]]).mean():.1f}장",
                            )
                        length_col_l = present_cols(low_df, ["지상부 길이"])
                        if length_col_l and len(low_df) >= 1:
                            st.metric(
                                "평균 지상부 길이",
                                f"{clean_nums(low_df[length_col_l[0]]).mean():.1f}mm",
                            )

                    with st.expander("📄 생육 결과 데이터 원본"):
                        st.dataframe(growth_df, width="stretch")
                        download_button(
                            growth_df,
                            "생육결과 CSV 다운로드",
                            "생육결과_download.csv",
                            key="download_growth_tab1",
                        )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ==============================================================
    # Tab 2: 지상부 길이 vs 잎 수
    # ==============================================================
    with tabs[1]:
        try:
            st.header("📏 지상부 길이와 잎의 갯수는 비례할까?")

            st.markdown("""
**산점도**에서 각 점은 하나의 개체를 나타내요 (x축: 지상부 길이, y축: 잎 수).  
점들이 우상향 직선 주변에 모여 있으면 두 변수가 함께 증가하는 **양적 상관**이 있다고 볼 수 있어요.

- **상관계수 r**: -1 ~ +1 사이 값. +1에 가까울수록 함께 증가, 0에 가까우면 관계가 약해요.
- **추세선(빨간 실선)**: np.polyfit으로 구한 직선 — 데이터의 전반적인 방향을 보여줘요.
            """.strip()

            )

            if growth_df.empty:
                st.info("생육 결과 CSV를 사이드바에서 올려주세요.")
            else:
                needed = ["지상부 길이", "잎 수"]
                missing = [c for c in needed if c not in growth_df.columns]
                if missing:
                    st.info(f"필요한 열이 없어요: {', '.join(missing)}")
                else:
                    fig, r_val = scatter_with_trend(
                        growth_df,
                        "지상부 길이",
                        "잎 수",
                        "지상부 길이 (mm)",
                        "잎 수 (장)",
                    )
                    st.plotly_chart(fig, width="stretch")

                    if r_val is not None:
                        direction = "양적" if r_val > 0 else "부적"
                        strength = (
                            "강함" if abs(r_val) >= 0.7
                            else ("중간" if abs(r_val) >= 0.4 else "약함")
                        )
                        slope_val, intercept_val = np.polyfit(
                            clean_nums(growth_df["지상부 길이"]),
                            clean_nums(growth_df["잎 수"]),
                            1,
                        )
                        st.markdown(f"""
**상관계수 r = {r_val:.3f}** → {direction} {strength} 상관

**추세선 식**: y = {slope_val:.3f} × x + {intercept_val:.3f}  
(지상부 길이가 1mm 증가할 때 잎 수는 약 {slope_val:.2f}장 함께 증가하는 경향)
                        """.strip())

                        if r_val >= 0.7:
                            st.markdown(
                                "✅ 지상부 길이와 잎 수 사이에 뚜렷한 양적 상관이 보여요 — "
                                "줄기가 길수록 잎도 많아지는 경향이 있어요."
                            )
                        elif r_val >= 0.4:
                            st.markdown(
                                "🤔 어느 정도 함께 움직이는 경향이 있지만, 예외도 꽤 있어요."
                            )
                        else:
                            st.markdown(
                                "❓ 뚜렷한 선형 관계가 보이지 않아요 — 길이 외에 다른 요인이 "
                                "잎 수에 더 큰 영향을 줄 수 있어요."
                            )

                    with st.expander("📄 생육 결과 데이터 원본"):
                        st.dataframe(growth_df, width="stretch")
                        download_button(
                            growth_df,
                            "생육결과 CSV 다운로드 (Tab2)",
                            "생육결과_download.csv",
                            key="download_growth_tab2",
                        )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ==============================================================
    # Tab 3: 지하부 무게 vs 잎 수
    # ==============================================================
    with tabs[2]:
        try:
            st.header("🍃 지하부 무게와 잎의 갯수는 비례할까?")

            st.markdown("""
**산점도**에서 각 점은 하나의 개체를 나타내요 (x축: 지하부 무게, y축: 잎 수).  
점들이 우상향 직선 주변에 모여 있으면 지하부 무게가 무거울수록 잎도 많아지는 **양적 상관**이 있다고 볼 수 있어요.

- **상관계수 r**: -1 ~ +1 사이 값. +1에 가까울수록 함께 증가, 0에 가까우면 관계가 약해요.
- **추세선(빨간 실선)**: np.polyfit으로 구한 직선이에요.
            """.strip()

            )

            if growth_df.empty:
                st.info("생육 결과 CSV를 사이드바에서 올려주세요.")
            else:
                needed = ["지하부 질량", "잎 수"]
                missing = [c for c in needed if c not in growth_df.columns]
                if missing:
                    st.info(f"필요한 열이 없어요: {', '.join(missing)}")
                else:
                    fig, r_val = scatter_with_trend(
                        growth_df,
                        "지하부 질량",
                        "잎 수",
                        "지하부 무게 (g)",
                        "잎 수 (장)",
                    )
                    st.plotly_chart(fig, width="stretch")

                    if r_val is not None:
                        direction = "양적" if r_val > 0 else "부적"
                        strength = (
                            "강함" if abs(r_val) >= 0.7
                            else ("중간" if abs(r_val) >= 0.4 else "약함")
                        )
                        slope_val2, intercept_val2 = np.polyfit(
                            clean_nums(growth_df["지하부 질량"]),
                            clean_nums(growth_df["잎 수"]),
                            1,
                        )
                        st.markdown(f"""
**상관계수 r = {r_val:.3f}** → {direction} {strength} 상관

**추세선 식**: y = {slope_val2:.3f} × x + {intercept_val2:.3f}  
(지하부 무게가 1g 증가할 때 잎 수는 약 {slope_val2:.2f}장 함께 증가하는 경향)
                        """.strip())

                        if r_val >= 0.7:
                            st.markdown(
                                "✅ 지하부 무게와 잎 수 사이에 뚜렷한 양적 상관이 보여요 — "
                                "뿌리가 무거울수록 잎도 많아지는 경향이 있어요."
                            )
                        elif r_val >= 0.4:
                            st.markdown(
                                "🤔 어느 정도 함께 움직이는 경향이 있지만, 예외도 꽤 있어요."
                            )
                        else:
                            st.markdown(
                                "❓ 뚜렷한 선형 관계가 보이지 않아요 — 지하부 무게보다 다른 요인이 "
                                "잎 수에 더 큰 영향을 줄 수 있어요."
                            )

                        # 추가: 지상부 무게 vs 잎 수 (비교용 작은 섹션)
                        if "지상부 질량" in growth_df.columns:
                            st.markdown("### 📎 참고: 지상부 무게 vs 잎 수")
                            fig2, r_val2 = scatter_with_trend(
                                growth_df,
                                "지상부 질량",
                                "잎 수",
                                "지상부 무게 (g)",
                                "잎 수 (장)",
                            )
                            st.plotly_chart(fig2, width="stretch")
                            if r_val2 is not None:
                                st.markdown(
                                    f"**지상부 무게 vs 잎 수 상관계수 r = {r_val2:.3f}**"
                                )
                                if abs(r_val2) > abs(r_val):
                                    st.markdown(
                                        "👉 지상부 무게가 지하부 무게보다 잎 수와 더 강한 상관을 보여요."
                                    )
                                elif abs(r_val) > abs(r_val2):
                                    st.markdown(
                                        "👉 지하부 무게가 지상부 무게보다 잎 수와 더 강한 상관을 보여요."
                                    )
                                else:
                                    st.markdown(
                                        "👉 두 무게가 잎 수와의 상관 정도가 비슷해요."
                                    )

                    with st.expander("📄 생육 결과 데이터 원본"):
                        st.dataframe(growth_df, width="stretch")
                        download_button(
                            growth_df,
                            "생육결과 CSV 다운로드 (Tab3)",
                            "생육결과_download.csv",
                            key="download_growth_tab3",
                        )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


if __name__ == "__main__":
    main()
