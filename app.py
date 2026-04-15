import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 대시보드 설정
st.set_page_config(layout="wide")
st.title("교육 역량 분석: 학업과 적성이 연봉에 미치는 영향")
st.write("본 대시보드는 데이터 표준화(Scaling)를 적용하여 각 역량의 상대적 기여도를 분석합니다.")

# 사이드바 - 파일 업로드
st.sidebar.header("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("career_data.csv 파일 선택", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['salary']) # 연봉 결측치 제거
    
    # 분석용 변수 설정
    features = ['ssc_p', 'degree_p', 'etest_p']
    target = 'salary'
    
    X = df[features]
    y = df[target]

    # 데이터 표준화 작업
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # 메뉴 구성
    st.sidebar.header("분석 단계")
    menu = st.sidebar.radio("단계 선택", ["1. 데이터 탐색 및 표준화", "2. 표준화 회귀분석 결과", "3. 최종 분석 결론 및 시사점"])

    if menu == "1. 데이터 탐색 및 표준화":
        st.header("1. 데이터 탐색 및 단위 표준화")
        
        # [추가] 영문 약자 설명 (변수 사전)
        st.subheader("📝 변수 정의 (Variable Dictionary)")
        dict_data = {
            "영문 약자": ["ssc_p", "degree_p", "etest_p", "salary"],
            "의미": ["중등 학업 성취도", "대학 학사 성적", "직무 적성 검사 점수", "최종 취업 연봉"],
            "상세 설명": [
                "10학년(고1)까지의 기초 학업 역량 및 성실성 지표",
                "대학 전공 학점 및 학술적 전문성 지표",
                "실무 문제 해결 능력 및 기업 요구 적성 점수",
                "노동 시장에서 평가된 학생의 경제적 가치(결과값)"
            ]
        }
        st.table(pd.DataFrame(dict_data))

        st.subheader("① 변수 간 상관관계 (히트맵)")
        # 히트맵 사이즈 조절 (figsize 축소 및 컨테이너 너비 고정 해제)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap="Oranges", ax=ax, fmt=".2f")
        st.pyplot(fig, use_container_width=False)
        
        st.divider()
        st.subheader("② 변수별 스케일 비교")
        st.info("성적(0~100)과 연봉(대단위)의 척도 차이를 해결하기 위해 표준화를 실시했습니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**[표준화 전 원본]**")
            st.dataframe(X.join(y).head())
        with col2:
            st.markdown("**[표준화 후 데이터]**")
            st.dataframe(X_scaled_df.head())

    elif menu == "2. 표준화 회귀분석 결과":
        st.header("2. 전체 데이터 기반 다중선형 회귀분석")
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        # [해설 추가] 모델 설명력
        st.subheader("📊 모델 전체 설명력 (R-squared)")
        st.metric(label="R-squared (결정계수)", value=f"{r2:.2f}")
        st.markdown(f"""
        **[수치 해설]**
        * 결정계수가 **{r2:.2f}**라는 것은, 우리가 설정한 3가지 역량이 연봉 차이의 약 **{max(0, r2*100):.1f}%**를 설명한다는 뜻입니다.
        * 이 수치가 낮을수록, 연봉 결정에는 성적 외의 '비인지적 요인(인성, 인맥, 면접 등)'이 더 크게 작용함을 시사합니다.
        """)
        
        st.divider()

        # [해설 추가] 회귀계수 가중치
        st.subheader("🎯 역량별 연봉 기여도 (가중치)")
        coef_df = pd.DataFrame({
            '역량 항목': ['중등 성적(ssc_p)', '대학 성적(degree_p)', '직무 적성(etest_p)'], 
            '영향력(표준화 계수)': model.coef_
        })
        st.table(coef_df)
        st.markdown("""
        **[수치 해설]**
        * **표준화 계수(Standardized Coefficient):** 모든 변수의 단위를 통일했으므로, 숫자 크기가 클수록 연봉에 미치는 영향력이 더 강력함을 의미합니다.
        * **양수(+):** 해당 역량이 높을수록 연봉이 상승하는 경향이 있습니다.
        * **음수(-):** 해당 변수가 다른 변수들과 복합적으로 작용할 때 나타나는 상대적 하락 요인 또는 통계적 노이즈를 의미합니다.
        """)

    elif menu == "3. 최종 분석 결론 및 시사점":
        st.header("3. 최종 분석 결론 및 시사점")
        
        model = LinearRegression().fit(X_scaled, y)
        importance = pd.Series(model.coef_, index=['중등 성적', '대학 성적', '직무 적성'])
        main_var = importance.idxmax()
        
        st.subheader("💡 1. 정량적 지표 간의 상대적 우위")
        st.info(f"분석 결과, 3가지 지표 중 **{main_var}**의 기여도가 상대적으로 가장 높았습니다. 이는 학업 성적보다는 실무 적성이 시장 가치와 더 밀접함을 보여줍니다.")
        
        st.subheader("🔍 2. 정량 평가의 '절대적 한계' 발견")
        st.warning("전체적인 설명력이 높지 않다는 점은,
