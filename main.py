# main.py
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
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
# 열 이름 표준화 (KeyError 방지)
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

    # 숫자 열만 숫자로 변환 시도 (글자 열은 그대로)
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

    # 생중량 열이 없고 지상부/지하부 질량이 모두 있으면 합산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # 시간 열 파싱 (두 가지 형식 대응)
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
# 학교명 추출 (파일명 앞부분에서) — 정규식 범위 오류 수정
def extract_school_name(filename):
    if not filename:
        return None
    base = filename.split(".")[0]          # 확장자 제거
    # '_', '-', '.'을 구분자로 분리 (하이픈은 이스케이프, 문자 범위 방지)
    parts = re.split(r"[_.\-]+", base)
    # 첫 요소 중 숫자로 시작하지 않는 것을 학교명으로 사용
    for p in parts:
        p = p.strip()
        if p and not p[0].isdigit():
            return p
    # 모두 숫자면 첫 요소 반환
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
    df_t = pd.DataFrame({"x": xf.reset_index(drop=True), "y": yf.reset_index(drop=True)})
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
    """산점도 + 추세선 그래프를 반환. 데이터가 부족하면 빈 그래프 + 안내."""
    r, slope, intercept, x_line, y_line = correlation_with_trend(
        df[x_col], df[y_col]
    )
    if r is None:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="데이터가 부족하여 산점도를 표시할 수 없어요",
            showarrow=False, font=dict(size=14),
        )
        set_fig_font(fig)
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=400,
        )
        return fig, None

    fig = make_subplots(rows=1, cols=1)

    # 산점도
    scatter = go.Scatter(
        x=df[x_col].dropna(),
        y=df[y_col].dropna(),
        mode="markers",
        marker=dict(size=8, opacity=0.7),
        name="개체",
        hoverinfo="x+y",
    )
    fig.add_trace(scatter, row=1, col=1)

    # 추세선
    trend = go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        line=dict(color="red", width=2),
        name=f"추세선 (r={r:.3f})",
    )
    fig.add_trace(trend, row=1, col=1)

    set_fig_font(fig)
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig, r


# ----------------------------------------------------------------------
# 메타데이터 요약 (학교명, 기간 등)
def get_env_summary(df):
    summary = {}
    if df.empty or "time" not in df.columns:
        return summary
    times = pd.to_datetime(df["time"], errors="coerce").dropna()
    if len(times) == 0:
        return summary
    school = extract_school_name(None)
    summary["학교명"] = school if school else "올린 파일 기준"
    summary["측정 기간"] = f"{times.min().strftime('%Y-%m-%d')} ~ {times.max().strftime('%Y-%m-%d')}"
    summary["데이터 행 수"] = len(df)
    return summary


def get_growth_summary(df):
    summary = {}
    if df.empty:
        return summary
    summary["개체 수"] = len(df)
    num_cols = present_cols(df, ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"])
    summary["측정 항목"] = ", ".join(num_cols) if num_cols else "없음"
    return summary


# ----------------------------------------------------------------------
# 메인 앱
def main():
    st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
    st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

    # 사이드바: CSV 업로드
    st.sidebar.header("📁 데이터 업로드")
    env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 CSV 파일을 올리면 자동으로 분석됩니다.\n"
        "인코딩(utf-8-sig/cp949)과 열 이름은 자동으로 처리돼요."
    )

    # 데이터 미리 읽기 (한 번만)
    env_raw = safe_read(env_file)
    growth_raw = safe_read(growth_file)

    env_df = std_cols(env_raw) if not env_raw.empty else pd.DataFrame()
    growth_df = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

    school_name = extract_school_name(
        env_file.name if env_file else (growth_file.name if growth_file else None)
    )

    tabs = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

    # --------------------------------------------------------------
    # Tab 1: 실험 개요
    # --------------------------------------------------------------
    with tabs[0]:
        try:
            st.header("📖 실험 개요")

            if school_name:
                st.subheader(f"🏫 {school_name} 극지식물 생육 분석")

            # 연구 배경 및 목적
            st.markdown(
                """
**연구 배경**
극지식물은 harsh한 환경에서도 살아남는 강인한 식물이에요.
이 대시보드에서는 우리 학교 극지식물 개체마다 생육 상태가 어떻게 다른지 살펴봅니다.

**분석 목적**
- 개체별 생중량 분포와 차이를 평균·표준편차·변동계수(CV)로 파악
- 잎 수·지상부 길이와 생중량의 관계를 상관분석으로 확인
- 생중량 상위 25% vs 하위 25% 개체를 비교
                """.strip()
            )

            # 데이터 요약 표
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 환경 데이터 요약")
                env_summary = get_env_summary(env_df)
                if env_summary:
                    st.table(
                        pd.DataFrame(
                            {"항목": list(env_summary.keys()), "값": list(env_summary.values())}
                        )
                    )
                else:
                    st.info("환경 CSV를 올리면 요약이 표시돼요.")

            with col2:
                st.markdown("### 생육 결과 데이터 요약")
                growth_summary = get_growth_summary(growth_df)
                if growth_summary:
                    st.table(
                        pd.DataFrame(
                            {"항목": list(growth_summary.keys()), "값": list(growth_summary.values())}
                        )
                    )
                else:
                    st.info("생육 CSV를 올리면 요약이 표시돼요.")

            # 주요 지표 카드 4개
            st.markdown("### 📊 주요 지표")
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                if not growth_df.empty:
                    n = len(growth_df)
                    st.metric("총 개체수", f"{n}개체")
                else:
                    st.metric("총 개체수", "—")

            with m2:
                if "temperature" in env_df.columns:
                    tmp = clean_nums(env_df["temperature"])
                    if len(tmp) >= 2:
                        st.metric("평균 온도", f"{tmp.mean():.1f}°C")
                    else:
                        st.metric("평균 온도", "데이터 부족")
                else:
                    st.metric("평균 온도", "—")

            with m3:
                if "humidity" in env_df.columns:
                    hum = clean_nums(env_df["humidity"])
                    if len(hum) >= 2:
                        st.metric("평균 습도", f"{hum.mean():.1f}%")
                    else:
                        st.metric("평균 습도", "데이터 부족")
                else:
                    st.metric("평균 습도", "—")

            with m4:
                if "생중량" in growth_df.columns:
                    gw = clean_nums(growth_df["생중량"])
                    if len(gw) >= 2:
                        st.metric("평균 생중량", f"{gw.mean():.2f}g")
                    else:
                        st.metric("평균 생중량", "데이터 부족")
                else:
                    st.metric("평균 생중량", "—")

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # --------------------------------------------------------------
    # Tab 2: 환경 데이터
    # --------------------------------------------------------------
    with tabs[1]:
        try:
            st.header("🌡️ 환경 데이터")

            if env_df.empty:
                st.info("환경 CSV를 사이드바에서 올려주세요.")
            else:
                env_cols = present_cols(
                    env_df, ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                )

                if not env_cols:
                    st.info("인식 가능한 환경 열이 없어요. 열 이름을 확인해주세요.")
                else:
                    # 환경 요약 (평균·최소·최대 표)
                    st.markdown("### 환경 요약 (평균·최소·최대)")
                    rows = []
                    for c in env_cols:
                        s = clean_nums(env_df[c])
                        if len(s) >= 1:
                            label = c.replace("_", " ").title()
                            rows.append(
                                {
                                    "측정 항목": label,
                                    "평균": round(s.mean(), 2),
                                    "최소": round(s.min(), 2),
                                    "최대": round(s.max(), 2),
                                }
                            )
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                    else:
                        st.info("환경 수치 데이터가 없어요.")

                    # 날짜별 시계열
                    st.markdown("### 날짜별 시계열 (일별 평균)")
                    if "time" in env_df.columns:
                        env_df_copy = env_df.copy()
                        env_df_copy["date"] = pd.to_datetime(
                            env_df_copy["time"], errors="coerce"
                        ).dt.date

                        daily_cols = [c for c in env_cols if c in env_df_copy.columns]
                        if daily_cols:
                            daily = (
                                env_df_copy.groupby("date")[daily_cols]
                                .mean()
                                .reset_index()
                            )

                            for c in daily_cols:
                                label = c.replace("_", " ").title()
                                unit = ""
                                if c == "temperature":
                                    unit = "°C"
                                elif c == "humidity":
                                    unit = "%"
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=daily["date"],
                                        y=daily[c],
                                        mode="lines+markers",
                                        name=f"{label}",
                                        line=dict(width=2),
                                    )
                                )
                                fig.update_layout(
                                    title=f"{label} 변화 ({unit})",
                                    xaxis_title="날짜",
                                    yaxis_title=f"{label} {unit}",
                                    height=280,
                                )
                                set_fig_font(fig)
                                st.plotly_chart(fig, width="stretch")
                        else:
                            st.info("일별 평균을 계산할 환경 열이 없어요.")
                    else:
                        st.info("시간(time) 열이 없어요.")

                    # expander: 원본 테이블 + CSV 다운로드
                    with st.expander("📄 환경 데이터 원본"):
                        st.dataframe(env_df, width="stretch")
                        csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="CSV 다운로드",
                            data=csv_bytes,
                            file_name="환경데이터_download.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # --------------------------------------------------------------
    # Tab 3: 생육 결과
    # --------------------------------------------------------------
    with tabs[2]:
        try:
            st.header("📊 생육 결과")

            if growth_df.empty:
                st.info("생육 결과 CSV를 사이드바에서 올려주세요.")
            else:
                growth_num_cols = present_cols(
                    growth_df,
                    ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"],
                )

                if not growth_num_cols:
                    st.info("인식 가능한 생육 수치 열이 없어요.")
                else:
                    # 🥇 핵심 결과 카드
                    st.markdown("### 🥇 핵심 결과")
                    k1, k2, k3 = st.columns(3)

                    if "생중량" in growth_df.columns:
                        gw = clean_nums(growth_df["생중량"])
                        if len(gw) >= 2:
                            mean_gw = gw.mean()
                            k1.metric("평균 생중량", f"{mean_gw:.2f}g")
                            idx_max = gw.idxmax()
                            idx_min = gw.idxmin()
                            max_label = growth_df.loc[idx_max, "개체번호"] if "개체번호" in growth_df.columns else "?"
                            min_label = growth_df.loc[idx_min, "개체번호"] if "개체번호" in growth_df.columns else "?"
                            k2.metric("생중량 최대 개체", f"{max_label} ({gw.max():.2f}g)")
                            k3.metric("생중량 최소 개체", f"{min_label} ({gw.min():.2f}g)")
                        else:
                            k1.metric("평균 생중량", "데이터 부족")
                            k2.metric("생중량 최대 개체", "—")
                            k3.metric("생중량 최소 개체", "—")
                    else:
                        k1.metric("평균 생중량", "열 없음")
                        k2.metric("생중량 최대 개체", "—")
                        k3.metric("생중량 최소 개체", "—")

                    # 개체별 생육 비교 (막대 그래프)
                    st.markdown("### 📈 개체별 생육 비교")

                    bar_cols = present_cols(
                        growth_df, ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                    )
                    if bar_cols:
                        bar_tab = st.tabs(
                            ["평균 생중량 ⭐", "평균 잎 수", "평균 지상부 길이", "평균 지하부 길이"]
                        )
                        bar_targets = {
                            "평균 생중량 ⭐": "생중량",
                            "평균 잎 수": "잎 수",
                            "평균 지상부 길이": "지상부 길이",
                            "평균 지하부 길이": "지하부 길이",
                        }

                        for i, (tab_label, col_name) in enumerate(bar_targets.items()):
                            with bar_tab[i]:
                                if col_name in growth_df.columns:
                                    s = clean_nums(growth_df[col_name])
                                    if len(s) >= 1:
                                        color = (
                                            "gold"
                                            if col_name == "생중량"
                                            else "skyblue"
                                        )
                                        fig = go.Figure()
                                        if "개체번호" in growth_df.columns:
                                            labels = growth_df["개체번호"].astype(str).tolist()
                                        else:
                                            labels = [str(i + 1) for i in range(len(s))]

                                        fig.add_trace(
                                            go.Bar(
                                                x=labels,
                                                y=s.values,
                                                marker_color=color,
                                                name=col_name,
                                                hoverinfo="x+y",
                                            )
                                        )
                                        set_fig_font(fig)
                                        fig.update_layout(
                                            xaxis_title="개체번호",
                                            yaxis_title=f"{col_name}",
                                            height=320,
                                            showlegend=False,
                                        )
                                        st.plotly_chart(fig, width="stretch")
                                    else:
                                        st.info(f"{col_name} 데이터가 부족해요.")
                                else:
                                    st.info(f"{col_name} 열이 없어요.")
                    else:
                        st.info("막대 그래프를 그릴 생육 열이 없어요.")

                    # 생중량 분포 (박스플롯)
                    st.markdown("### 📦 생중량 분포")
                    if "생중량" in growth_df.columns:
                        gw = clean_nums(growth_df["생중량"])
                        if len(gw) >= 2:
                            fig = go.Figure()
                            q75 = gw.quantile(0.75)
                            q25 = gw.quantile(0.25)
                            if "개체번호" in growth_df.columns:
                                labels = growth_df["개체번호"].astype(str).tolist()
                            else:
                                labels = [str(i + 1) for i in range(len(gw))]

                            fig.add_trace(
                                go.Box(
                                    y=gw.values,
                                    name="생중량",
                                    fillcolor="skyblue",
                                    boxpoints="outliers",
                                    hoverinfo="y",
                                )
                            )
                            set_fig_font(fig)
                            fig.update_layout(
                                yaxis_title="생중량 (g)",
                                height=300,
                                showlegend=False,
                            )
                            st.plotly_chart(fig, width="stretch")

                            top_idx = (gw >= q75).sum()
                            low_idx = (gw <= q25).sum()
                            st.markdown(
                                f"""
**상위 25% (≥ Q75 = {q75:.2f}g)**: {top_idx}개체  
**하위 25% (≤ Q25 = {q25:.2f}g)**: {low_idx}개체
                                """
                            )
                        else:
                            st.info("생중량 데이터가 부족해요.")
                    else:
                        st.info("생중량 열이 없어요.")

                    # 상관관계 분석 (산점도 2개 + 추세선)
                    st.markdown("### 🔗 상관관계 분석")
                    st.markdown(
                        """
- **상관계수(r)**: -1 ~ +1 사이의 값.  
  +1에 가까울수록 함께 증가(양적 상관), -1에 가까울수록 한 쪽이 증가할 때 다른 쪽이 감소(부적 상관), 0에 가까우면 관계가 약해요.
- **추세선**: 빨간 실선은 데이터의 전반적인 방향을 보여줘요.
                        """.strip()
                    )

                    corr_tab = st.tabs(["🥬 잎 수 vs 생중량", "📏 지상부 길이 vs 생중량"])
                    scatter_pairs = [
                        ("잎 수", "생중량"),
                        ("지상부 길이", "생중량"),
                    ]

                    for i, (x_col, y_col) in enumerate(scatter_pairs):
                        with corr_tab[i]:
                            if x_col in growth_df.columns and y_col in growth_df.columns:
                                fig, r_val = scatter_with_trend(
                                    growth_df,
                                    x_col,
                                    y_col,
                                    x_label=x_col,
                                    y_label=y_col,
                                )
                                st.plotly_chart(fig, width="stretch")
                                if r_val is not None:
                                    direction = "양적" if r_val > 0 else "부적"
                                    strength = (
                                        "강함"
                                        if abs(r_val) >= 0.7
                                        else ("중간" if abs(r_val) >= 0.4 else "약함")
                                    )
                                    st.markdown(
                                        f"**상관계수 r = {r_val:.3f}** → {direction} {strength} 상관"
                                    )
                                else:
                                    st.info("상관계수를 계산할 데이터가 부족해요.")
                            else:
                                missing = [c for c in [x_col, y_col] if c not in growth_df.columns]
                                st.info(f"필요한 열이 없어요: {', '.join(missing)}")

                    # expander: 생육 데이터 원본 + CSV 다운로드
                    with st.expander("📄 생육 데이터 원본"):
                        st.dataframe(growth_df, width="stretch")
                        csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="CSV 다운로드",
                            data=csv_bytes,
                            file_name="생육결과_download.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


if __name__ == "__main__":
    main()
