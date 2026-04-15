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
st.write("데이터 표준화(Scaling)를 적용하고, 전체 데이터를 기반으로 다중선형 회귀분석을 수행합니다.")

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    st.sidebar.header("분석 단계")
    menu = st.sidebar.radio("단계 선택", ["1. 데이터 탐색 및 표준화", "2. 표준화 회귀분석 결과", "3. 최종 분석 결론"])

    if menu == "1. 데이터 탐색 및 표준화":
        st.header("1. 데이터 탐색 (EDA) 및 단위 표준화")
        
        st.subheader("① 변수 간 상관관계 (히트맵)")
        st.write("각 역량 변수와 연봉 간의 1차적인 상관관계를 시각적으로 확인합니다.")
        # 그래프 생성 코드 (부활)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap="Oranges", ax=ax, fmt=".2f")
        st.pyplot(fig)
        
        st.divider()
        
        st.subheader("② 변수별 스케일 비교 (표준화 전 vs 후)")
        st.info("교수님 자문 반영: 점수(0~100)와 연봉(대단위)의 척도 차이로 인한 왜곡을 막기 위해 데이터를 표준화했습니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**[표준화 전 Raw Data]**")
            st.dataframe(X.join(y).head())
        with col2:
            st.markdown("**[표준화 후 Scaled Data] (평균 0 중심)**")
            st.dataframe(X_scaled_df.head())

    elif menu == "2. 표준화 회귀분석 결과":
        st.header("2. 전체 데이터 기반 다중선형 회귀분석")
        
        # 전체 데이터로 모델 훈련 (Train/Test 분리 없음)
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        st.success("모델 훈련 완료 (전체 데이터 100% 사용)")
        st.metric(label="모델 전체 설명력 (R-squared)", value=f"{r2:.2f}")
        
        coef_df = pd.DataFrame({'역량 항목': ['중등 성적(ssc_p)', '대학 성적(degree_p)', '직무 적성(etest_p)'], '영향력(표준화 계수)': model.coef_})
        st.subheader("각 역량별 연봉 기여도 (가중치)")
        st.table(coef_df)
        st.caption("※ 음수(-) 계수는 단위 왜곡이 아닌, 모델 내에서의 상대적 역상관관계를 의미합니다.")

    elif menu == "3. 최종 분석 결론":
        st.header("3. 교수님 보고용 시사점")
        
        model = LinearRegression().fit(X_scaled, y)
        importance = pd.Series(model.coef_, index=['중등 성적', '대학 성적', '직무 적성'])
        main_var = importance.idxmax()
        
        st.write("표준화 모델 분석 결과, 계수의 크기를 동일한 선상에서 객관적으로 비교할 수 있게 되었습니다.")
        st.success(f"최종적으로 **{main_var}** 변수가 연봉 결정에 가장 긍정적인 영향을 미치는 주요 인자로 도출되었습니다.")
        st.write("이러한 분석 결과는, 우리가 측정하고 있는 기존의 학업 지표와 실제 시장의 보상 체계 사이에 존재하는 구조적 차이를 여실히 보여줍니다.")

else:
    st.warning("왼쪽 사이드바에서 파일을 업로드해 주십시오.")
