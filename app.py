import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 대시보드 설정
st.set_page_config(layout="wide")
st.title("교육 역량 분석 (표준화 모델): 학업과 적성이 연봉에 미치는 영향")
st.write("교수님 자문에 따라 데이터 표준화(Scaling)를 적용하고 전체 데이터를 기반으로 분석을 재구성했습니다.")

# 사이드바 - 파일 업로드
st.sidebar.header("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("career_data.csv 파일 선택", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['salary']) # 연봉 결측치 제거
    
    # 분석용 변수 추출
    features = ['ssc_p', 'degree_p', 'etest_p']
    target = 'salary'
    
    X = df[features]
    y = df[target]

    # [핵심] 데이터 표준화 (Standardization)
    # 단위가 다른 변수들을 동일한 척도로 변환합니다.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    st.sidebar.header("분석 단계")
    menu = st.sidebar.radio("단계 선택", ["1. 데이터 분포 확인", "2. 표준화 회귀분석 결과", "3. 교수님 보고용 시사점"])

    if menu == "1. 데이터 분포 확인":
        st.header("1. 변수별 스케일 비교")
        st.write("표준화 전(Raw) 데이터와 표준화 후(Scaled) 데이터를 비교합니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("표준화 전 (0~100점 vs 연봉)")
            st.dataframe(X.join(y).head())
        with col2:
            st.subheader("표준화 후 (평균 0 중심의 상대 점수)")
            st.dataframe(X_scaled_df.head())

    elif menu == "2. 표준화 회귀분석 결과":
        st.header("2. 전체 데이터 기반 다중선형 회귀분석")
        
        # 전체 데이터로 모델 훈련 (Train/Test 분리 없음)
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        st.success("모델 분석 완료 (데이터 100% 사용)")
        st.metric(label="모델 전체 설명력 (R-squared)", value=f"{r2:.2f}")
        
        # 가중치(회귀계수) 분석
        coef_df = pd.DataFrame({'역량 항목': ['중등 성적', '대학 성적', '직무 적성'], '영향력(표준화 계수)': model.coef_})
        st.subheader("각 역량별 연봉 기여도 (가중치)")
        st.table(coef_df)
        
        st.info("※ 표준화 계수가 양수(+)이면 연봉 상승 요인, 음수(-)이면 상대적 하락 요인을 의미합니다.")

    elif menu == "3. 교수님 보고용 시사점":
        st.header("3. 최종 분석 결론")
        st.write("표준화 모델을 통해 도출된 데이터 기반 인사이트입니다.")
        
        # 가장 영향력 큰 변수 찾기 (절댓값 기준)
        model = LinearRegression().fit(X_scaled, y)
        importance = pd.Series(model.coef_, index=['중등 성적', '대학 성적', '직무 적성'])
        main_var = importance.idxmax()
        
        st.success(f"최종적으로 **{main_var}** 항목이 연봉 결정에 가장 지배적인 영향력을 행사하는 것으로 확인되었습니다.")
        st.write("단위 불일치 문제를 표준화로 해결함으로써, 각 역량이 가진 '순수한 기여도'를 파악할 수 있게 되었습니다.")

else:
    st.warning("왼쪽 사이드바에서 파일을 업로드해 주십시오.")
