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

# 2. 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv("career_data.csv")
    # 결측치 처리: 연봉(salary)이 없는 데이터 삭제
    df = df.dropna(subset=['salary'])
    return df

try:
    df = load_data()
    
    st.sidebar.header("분석 파이프라인")
    menu = st.sidebar.radio("단계 선택", ["1. 데이터 탐색 (EDA)", "2. 다중선형 회귀분석", "3. 결론 및 교육적 시사점"])

    if menu == "1. 데이터 탐색 (EDA)":
        st.header("1. 탐색적 데이터 분석 (EDA)")
        st.write("결측치가 제거된 정제된 데이터의 상위 5개 행입니다.")
        st.dataframe(df.head())
        
        st.subheader("변수 간 상관관계 (Heatmap)")
        # 분석에 사용할 숫자 데이터만 추출
        numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'salary']
        numeric_df = df[numeric_cols]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Oranges", ax=ax, fmt=".2f")
        st.pyplot(fig)
        st.info("변수 설명: ssc_p(중/고교성적), hsc_p(고교졸업성적), degree_p(대학성적), etest_p(적성검사), salary(연봉)")

    elif menu == "2. 다중선형 회귀분석":
        st.header("2. 기계학습 모델 훈련")
        st.write("학업 성취 및 적성 점수(독립변수)가 연봉(종속변수)에 미치는 영향을 예측합니다.")
        
        # 변수 설정
        X = df[['ssc_p', 'degree_p', 'etest_p']] # 독립변수 (고교성적, 대학성적, 적성검사)
        y = df['salary'] # 종속변수 (연봉)
        
        # 훈련용(80%)/테스트용(20%) 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 모델 훈련
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 예측 및 평가
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
            
            st.subheader("독립변수별 회귀 계수 (가중치)")
            coef_df = pd.DataFrame({'변수명': cols, '영향력(계수)': model.coef_})
            st.dataframe(coef_df)
            
            st.write("### 📌 교육적 시사점")
            st.write("- **데이터 해석:** 각 독립변수의 계수(숫자)가 높을수록 연봉 상승에 미치는 긍정적 영향력이 큼을 의미합니다.")
            st.write("- **현장 적용:** 막연한 진로 지도 대신, 데이터에 기반하여 학생들에게 '어떤 역량'을 우선적으로 개발해야 하는지 객관적인 지표로 상담할 수 있습니다.")
        else:
            st.warning("이전 탭에서 모델을 먼저 훈련시켜 주세요.")

except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. GitHub에 'career_data.csv'가 있는지 확인하세요.")
