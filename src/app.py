import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from recommender import CTWPFRecommender


# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="CTWP 강의 추천", layout="wide")

st.title("🎓Persona-Bridge")
st.title("학생 페르소나 맞춤형 학점교류 추천시스템")
st.markdown("""
당신의 역량 지식 그래프를 분석하여, 교류 대학의 커리큘럼 중 당신에게 필요한 융합 강의를 연결해 드립니다
""")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

if not st.session_state["GEMINI_API_KEY"]:
    user_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    if user_api_key:
        st.session_state["GEMINI_API_KEY"] = user_api_key

# --- [입력 폼] ---
st.subheader("📝 학생 페르소나 입력")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 1. 필수 정보")
    input_major = st.selectbox("주전공", ["사학과", "경영학과", "컴퓨터공학과", "전자공학과"])
    input_double = st.selectbox("복수전공", ["없음", "마케팅", "경영학과", "컴퓨터공학과"])

with col2:
    st.markdown("##### 2. 선택 정보")
    input_history = st.text_input("수강했던 전공 과목")
    # [요청 1] '/직무' 제거
    input_interest = st.text_input("관심있는 분야")

run_btn = st.button("🚀 교류 대학 강의 추천 시작", type="primary")

if run_btn:
    api_key = st.session_state["GEMINI_API_KEY"]
    
    if not api_key:
        st.error("⚠️ API Key가 필요합니다.")
    else:
        recommender = CTWPFRecommender(api_key)
        
        history_list = [h.strip() for h in input_history.split(',')] if input_history else []
        interest_val = input_interest if input_interest else None
        
        # [요청 2] 문구 수정
        with st.spinner("강의 추천 중..."):
            result_df, profile_keywords = recommender.run_analysis(
                input_major, input_double, history_list, interest_val
            )

        # 결과 출력
        st.divider()
        st.subheader(f"🏆 추천 강의 목록")
        
        if result_df.empty:
            st.warning("추천 결과가 없습니다. 입력 정보를 확인해주세요.")
        else:
            # 1위 강의 정보
            top_course = result_df.iloc[0]
            top_score_pct = top_course['적합도(%)']
            top_uni = top_course['university']
            
            # 최고 추천 강의 표시
            st.success(f"**가장 적합한 강의:** {top_course['강의명']} (적합도 : {top_score_pct}%)")
            st.caption(f"교수: {top_course['교수']} | 학교: {top_uni}")

            # 리스트 표시
            display_df = result_df.copy()
            display_df["강의 정보"] = display_df.apply(lambda x: f"{x['강의명']} - {x['교수']} ({x['university']})", axis=1)
            display_df["적합도"] = display_df["적합도(%)"].apply(lambda x: f"{x}%")
            
            # [요청 3] 인덱스 숨김 (hide_index=True)
            st.table(display_df[["강의 정보", "적합도"]].reset_index(drop=True))

        # [요청 4 & 5] 시각화 제거 및 근거 설명 창 추가
        with st.expander("📊 추천 근거", expanded=True):
            if not result_df.empty:
                st.markdown("### 지식 그래프 기반 강의 추천 근거")

                for i, row in result_df.iterrows():
                    st.markdown(f"**{row['강의명']}**")
                    
                    # Gemini가 생성한 '추천 사유' 출력
                    st.info(f"💡 {row['추천 사유']}")
                    
                    # CTWP 키워드 매칭 정보도 작게 표시
                    if row['매칭 키워드'] != "없음":
                        st.caption(f"🔗 매칭된 핵심 키워드: {row['매칭 키워드']}")
                    
                    st.markdown("---")