import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 대시보드 제목
st.set_page_config(layout="wide")
st.title("교육 데이터 분석: 진로 역량 및 학업 성취가 연봉에 미치는 영향")
st.write("다중선형 회귀분석을 통해 고교/대학 성적과 적성검사 점수가 실제 취업 시장(연봉)에서 갖는 가중치를 분석합니다.")

# ==========================================
# 🚨 변경된 핵심 포인트: 직접 파일 업로드 위젯
# ==========================================
st.sidebar.header("📁 데이터 업로드")
st.sidebar.info("교수님 지침에 따라 데이터는 서버에 저장하지 않습니다. 로컬 PC에 있는 CSV 파일을 아래에 업로드해 주세요.")
uploaded_file = st.sidebar.file_uploader("career_data.csv 파일 선택", type=['csv'])

if uploaded_file is not None:
    # 파일을 업로드했을 때만 분석 시작
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['salary']) # 결측치 처리
    
    st.sidebar.header("분석 파이프라인")
    menu = st.sidebar.radio("단계 선택", ["1. 데이터 탐색 (EDA)", "2. 다중선형 회귀분석", "3. 결론 및 교육적 시사점"])

    if menu == "1. 데이터 탐색 (EDA)":
        st.header("1. 탐색적 데이터 분석 (EDA)")
        st.dataframe(df.head())
        
        st.subheader("변수 간 상관관계 (Heatmap)")
        numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'salary']
        numeric_df = df[numeric_cols]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Oranges", ax=ax, fmt=".2f")
        st.pyplot(fig)

    elif menu == "2. 다중선형 회귀분석":
        st.header("2. 기계학습 모델 훈련")
        
        X = df[['ssc_p', 'degree_p', 'etest_p']] 
        y = df['salary'] 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        st.success("다중선형 회귀 모델 훈련 완료")
        st.metric(label="모델 설명력 (R-squared)", value=f"{r2:.2f}")
        
        st.session_state['model'] = model
        st.session_state['X_columns'] = ['고교 성적(ssc_p)', '대학 성적(degree_p)', '적성검사(etest_p)']

    elif menu == "3. 결론 및 교육적 시사점":
        st.header("3. 모델 해석 및 시사점 도출")
        if 'model' in st.session_state:
            model = st.session_state['model']
            cols = st.session_state['X_columns']
            
            coef_df = pd.DataFrame({'변수명': cols, '영향력(계수)': model.coef_})
            st.dataframe(coef_df)
            
            st.write("### 📌 교육적 시사점")
            st.write("- 각 독립변수의 계수(숫자)가 높을수록 연봉 상승에 미치는 긍정적 영향력이 큼을 의미합니다.")
        else:
            st.warning("이전 탭에서 모델을 먼저 훈련시켜 주세요.")

else:
    # 파일을 업로드하기 전 대기 화면
    st.warning("분석을 시작하려면 왼쪽 사이드바에서 데이터 파일(CSV)을 먼저 업로드해 주십시오.")
