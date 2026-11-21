import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ──────────────────────────────────────────────
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

# ── 상수 ────────────────────────────────────────────────────────
KOR_TAB1 = "탭1: 개체들마다 생중량 차이는 얼마나 날까?"
KOR_TAB2 = "탭2: 지상부 무게와 잎의 갯수는 비례할까?"
KOR_TAB3 = "탭3: 지하부 무게와 잎의 갯수는 비례할까?"

# ── CSV 읽기 (UTF-8-SIG → CP949 폴백) ──────────────────────────
@st.cache_data
def read_csv_cached(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            return df
        except Exception:
            pass
    return pd.DataFrame()


# ── 열 이름 표준화 ───────────────────────────────────────────────
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
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ── 산점도 + 추세선 (탭2·탭3 공용) ────────────────────────────
def scatter_with_trend(x, y, xlabel, ylabel, title, color="#4C72B0"):
    df_plot = pd.DataFrame({xlabel: x, ylabel: y}).dropna()
    if len(df_plot) < 2:
        return None

    x_vals = df_plot[xlabel].values
    y_vals = df_plot[ylabel].values
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    r = np.corrcoef(x_vals, y_vals)[0, 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(color=color, size=8, opacity=0.75),
            name=f"{ylabel}",
        )
    )
    y_fit = slope * x_vals + intercept
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_fit,
            mode="lines",
            line=dict(color="#E45756", width=2),
            name=f"추세선 (y={slope:.4f}x+{intercept:.4f})",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=400,
    )
    return fig, slope, intercept, r


# ── 사이드바 ─────────────────────────────────────────────────────
st.sidebar.title("📂 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

# 학교명 추출 (업로드 파일명 앞부분)
school_name = "우리 학교"
if growth_file is not None:
    fname = os.path.basename(growth_file.name)
    base = fname.split(".")[0]
    school_name = base

st.sidebar.markdown(f"### 🏫 학교: {school_name}")

# 생육 측정 항목 (사이드바)
growth_cols = [
    "잎 수",
    "지상부 길이",
    "지하부 길이",
    "지상부 질량",
    "지하부 질량",
    "생중량",
]
selected_cols = []
if growth_file is not None:
    growth_df_raw = read_csv_cached(growth_file)
    growth_df = std_cols(growth_df_raw)
    available = [c for c in growth_cols if c in growth_df.columns]
    selected_cols = st.sidebar.multiselect(
        "📊 생육 측정 항목 선택",
        available,
        default=available,
    )

# 데이터 로딩 (스피너)
with st.spinner("데이터 로딩 중..."):
    env_df = pd.DataFrame()
    if env_file is not None:
        env_df = std_cols(read_csv_cached(env_file))
    if growth_file is not None and not selected_cols:
        growth_df = std_cols(read_csv_cached(growth_file))

# ── 메인 화면 ─────────────────────────────────────────────────────
st.set_page_config(page_title="송도고 그린 3팀", layout="wide")
st.title("송도고등학교 그린 3팀 2026")

# 파일 없을 때 안내
if env_file is None and growth_file is None:
    st.info("📁 사이드바에서 환경 CSV와 생육 결과 CSV를 업로드해 주세요.")
    st.stop()

# ── 탭 ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([KOR_TAB1, KOR_TAB2, KOR_TAB3])

# ══════════════════════════════════════════════════════════════════
# 탭1: 생중량 분포 + 통계
# ══════════════════════════════════════════════════════════════════
with tab1:
    try:
        if growth_file is None:
            st.info("📁 생육 결과 CSV를 업로드해 주세요.")
        elif "생중량" not in growth_df.columns:
            st.info("⚠️ 생중량 열이 데이터에 없어요.")
        else:
            gw = growth_df["생중량"].dropna()
            if len(gw) < 2:
                st.info("⚠️ 생중량 데이터가 충분하지 않아요.")
            else:
                mean_val = gw.mean()
                std_val = gw.std()
                var_val = gw.var()
                max_val = gw.max()
                min_val = gw.min()
                max_id = growth_df.loc[gw.idxmax(), "개체번호"]
                min_id = growth_df.loc[gw.idxmin(), "개체번호"]
                diff_max = max_val - mean_val
                diff_min = mean_val - min_val

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("평균 생중량 (g)", f"{mean_val:.3f}")
                col2.metric("분산", f"{var_val:.4f}")
                col3.metric("표준편차 (g)", f"{std_val:.3f}")
                col4.metric("최대값", f"{max_val:.3f}")
                col5.metric("최소값", f"{min_val:.3f}")

                st.markdown("---")

                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=gw,
                        nbinsx=20,
                        marker_color="#4C72B0",
                        opacity=0.8,
                        name="생중량 분포",
                    )
                )
                fig_hist.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="#E45756",
                    line_width=2,
                    annotation_text=f"평균: {mean_val:.3f}g",
                    annotation_position="top right",
                )
                fig_hist.add_vline(
                    x=max_val,
                    line_dash="dot",
                    line_color="#2CA02C",
                    line_width=2,
                    annotation_text=f"최대: {max_val:.3f}g (개체 {max_id})",
                    annotation_position="top left",
                )
                fig_hist.add_vline(
                    x=min_val,
                    line_dash="dot",
                    line_color="#FF7F0E",
                    line_width=2,
                    annotation_text=f"최소: {min_val:.3f}g (개체 {min_id})",
                    annotation_position="bottom right",
                )
                fig_hist.update_layout(
                    title="개체별 생중량 분포 (히스토그램)",
                    xaxis_title="생중량 (g)",
                    yaxis_title="개체 수",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=380,
                )
                st.plotly_chart(fig_hist, width="stretch")

                st.info(
                    f"""
**결과 해석**
- 전체 {len(gw)}개체의 평균 생중량은 **{mean_val:.3f}g**, 표준편차는 **{std_val:.3f}g**입니다.
- 가장 큰 개체(개체 {max_id})는 {max_val:.3f}g으로 평균보다 **{diff_max:.3f}g** 더 무겁고,
  가장 작은 개체(개체 {min_id})는 {min_val:.3f}g으로 평균보다 **{diff_min:.3f}g** 더 가볍습니다.
- 분산 {var_val:.4f}로 개체 간 생중량 차이가 상당히 큰 편임을 알 수 있습니다.
"""
                )

                # 추가: 상위·하위 5개체 테이블
                st.markdown("---")
                st.markdown("### 상위 5개체 · 하위 5개체")
                gw_sorted = growth_df[["개체번호", "생중량"]].dropna().sort_values("생중량")
                top5 = gw_sorted.tail(5)
                bot5 = gw_sorted.head(5)
                st.dataframe(
                    top5.assign(**{"생중량 (g)": top5["생중량"].round(3)}).drop(columns=["생중량"]),
                    width="stretch",
                )
                st.write("")
                st.dataframe(
                    bot5.assign(**{"생중량 (g)": bot5["생중량"].round(3)}).drop(columns=["생중량"]),
                    width="stretch",
                )

                # 생중량 CSV 다운로드
                csv_bytes = (
                    growth_df[["개체번호", "생중량"]]
                    .dropna()
                    .to_csv(index=False)
                    .encode("utf-8-sig")
                )
                st.download_button(
                    "📥 생중량 데이터 다운로드 (CSV)",
                    csv_bytes,
                    file_name=f"{school_name}_생중량.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════════════════════════
# 탭2: 지상부 질량 vs 잎 수
# ══════════════════════════════════════════════════════════════════
with tab2:
    try:
        if growth_file is None:
            st.info("📁 생육 결과 CSV를 업로드해 주세요.")
        else:
            x_col = "지상부 질량"
            y_col = "잎 수"
            has_x = x_col in growth_df.columns
            has_y = y_col in growth_df.columns

            if not has_x or not has_y:
                missing = []
                if not has_x:
                    missing.append("지상부 질량")
                if not has_y:
                    missing.append("잎 수")
                st.info(f"⚠️ 다음 열이 데이터에 없어 그래프를 그릴 수 없어요: {', '.join(missing)}")
            else:
                fig_res = scatter_with_trend(
                    growth_df[x_col].dropna(),
                    growth_df[y_col].dropna(),
                    "지상부 질량 (g)",
                    "잎 수 (장)",
                    "지상부 질량 vs 잎 수 (산점도 + 추세선)",
                    color="#4C72B0",
                )
                if fig_res is None:
                    st.info("⚠️ 데이터가 2개 미만이에요.")
                else:
                    fig, slope, intercept, r = fig_res
                    st.plotly_chart(fig, width="stretch")

                    st.info(
                        f"""
**결과 해석**
- 지상부 질량과 잎 수 사이의 피어슨 상관계수는 **r = {r:.3f}**입니다.
- 추세선 기울기: {slope:.4f} → 지상부 질량이 1g 증가할 때 잎 수는 약 {slope:.2f}장 증가합니다.
- |r|이 1에 가까울수록 강한 비례 관계, 0에 가까울수록 약한 관계입니다.
  현재 r = {r:.3f}은 {'강한' if abs(r) > 0.7 else '중간' if abs(r) > 0.4 else '약한'} 상관관계를 나타냅니다.
"""
                    )

                    # 산점도 드론: 읍면동별 (개체번호 제외)
                    if "지상부 길이" in growth_df.columns:
                        fig2 = scatter_with_trend(
                            growth_df["지상부 길이"].dropna(),
                            growth_df[y_col].dropna(),
                            "지상부 길이 (mm)",
                            "잎 수 (장)",
                            "지상부 길이 vs 잎 수 (참고)",
                            color="#55A868",
                        )
                        if fig2 is not None:
                            st.markdown("---")
                            st.markdown("### 참고: 지상부 길이 vs 잎 수")
                            st.plotly_chart(fig2[0], width="stretch")

                    # 다운로드
                    csv_bytes = (
                        growth_df[[x_col, y_col]]
                        .dropna()
                        .to_csv(index=False)
                        .encode("utf-8-sig")
                    )
                    st.download_button(
                        "📥 상관 분석 데이터 다운로드 (CSV)",
                        csv_bytes,
                        file_name=f"{school_name}_지상부_vs_잎수.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════════════════════════
# 탭3: 지하부 질량 vs 잎 수
# ══════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_file is None:
            st.info("📁 생육 결과 CSV를 업로드해 주세요.")
        else:
            x_col = "지하부 질량"
            y_col = "잎 수"
            has_x = x_col in growth_df.columns
            has_y = y_col in growth_df.columns

            if not has_x or not has_y:
                missing = []
                if not has_x:
                    missing.append("지하부 질량")
                if not has_y:
                    missing.append("잎 수")
                st.info(f"⚠️ 다음 열이 데이터에 없어 그래프를 그릴 수 없어요: {', '.join(missing)}")
            else:
                fig_res = scatter_with_trend(
                    growth_df[x_col].dropna(),
                    growth_df[y_col].dropna(),
                    "지하부 질량 (g)",
                    "잎 수 (장)",
                    "지하부 질량 vs 잎 수 (산점도 + 추세선)",
                    color="#8172B3",
                )
                if fig_res is None:
                    st.info("⚠️ 데이터가 2개 미만이에요.")
                else:
                    fig, slope, intercept, r = fig_res
                    st.plotly_chart(fig, width="stretch")

                    st.info(
                        f"""
**결과 해석**
- 지하부 질량과 잎 수 사이의 피어슨 상관계수는 **r = {r:.3f}**입니다.
- 추세선 기울기: {slope:.4f} → 지하부 질량이 1g 증가할 때 잎 수는 약 {slope:.2f}장 증가합니다.
- |r|이 1에 가까울수록 강한 비례 관계, 0에 가까울수록 약한 관계입니다.
  현재 r = {r:.3f}은 {'강한' if abs(r) > 0.7 else '중간' if abs(r) > 0.4 else '약한'} 상관관계를 나타냅니다.
"""
                    )

                    # 참고: 지하부 길이 vs 잎 수
                    if "지하부 길이" in growth_df.columns:
                        fig2 = scatter_with_trend(
                            growth_df["지하부 길이"].dropna(),
                            growth_df[y_col].dropna(),
                            "지하부 길이 (mm)",
                            "잎 수 (장)",
                            "지하부 길이 vs 잎 수 (참고)",
                            color="#C44E52",
                        )
                        if fig2 is not None:
                            st.markdown("---")
                            st.markdown("### 참고: 지하부 길이 vs 잎 수")
                            st.plotly_chart(fig2[0], width="stretch")

                    # 다운로드
                    csv_bytes = (
                        growth_df[[x_col, y_col]]
                        .dropna()
                        .to_csv(index=False)
                        .encode("utf-8-sig")
                    )
                    st.download_button(
                        "📥 상관 분석 데이터 다운로드 (CSV)",
                        csv_bytes,
                        file_name=f"{school_name}_지하부_vs_잎수.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
