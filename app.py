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

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # 🚨 주의: 사이드바 메뉴 이름과 아래 if/elif 조건문 이름이 반드시 일치해야 합니다.
    st.sidebar.header("분석 단계")
    menu = st.sidebar.radio("단계 선택", ["1. 데이터 탐색 및 표준화", "2. 표준화 회귀분석 결과", "3. 최종 분석 결론 및 시사점"])

    if menu == "1. 데이터 탐색 및 표준화":
        st.header("1. 데이터 탐색 및 단위 표준화")
        
        st.subheader("① 변수 간 상관관계 (히트맵)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap="Oranges", ax=ax, fmt=".2f")
        st.pyplot(fig)
        
        st.divider()
        
        st.subheader("② 변수별 스케일 비교")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**[표준화 전 원본 데이터]**")
            st.dataframe(X.join(y).head())
        with col2:
            st.markdown("**[표준화 후 데이터] (평균 0 중심)**")
            st.dataframe(X_scaled_df.head())

    elif menu == "2. 표준화 회귀분석 결과":
        st.header("2. 전체 데이터 기반 다중선형 회귀분석")
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        st.success("분석 완료")
        st.metric(label="모델 전체 설명력 (R-squared)", value=f"{r2:.2f}")
        
        coef_df = pd.DataFrame({
            '역량 항목': ['중등 성적(ssc_p)', '대학 성적(degree_p)', '직무 적성(etest_p)'], 
            '영향력(표준화 계수)': model.coef_
        })
        st.subheader("각 역량별 연봉 기여도 (가중치)")
        st.table(coef_df)

    elif menu == "3. 최종 분석 결론 및 시사점":
        st.header("3. 최종 분석 결론 및 시사점")
        
        model = LinearRegression().fit(X_scaled, y)
        importance = pd.Series(model.coef_, index=['중등 성적', '대학 성적', '직무 적성'])
        main_var = importance.idxmax()
        
        st.subheader("💡 1. 정량적 지표 간의 상대적 우위")
        st.info(f"분석 결과, 3가지 지표 중 **{main_var}**의 계수가 가장 높게 나타났습니다. 이는 과거의 내신 성적보다는 실무 중심의 적성이 연봉에 미치는 영향력이 상대적으로 더 큼을 의미합니다.")
        
        st.subheader("🔍 2. 정량 평가의 '절대적 한계' 발견")
        st.warning("하지만 전체 모델의 설명력이 낮다는 점에 주목해야 합니다. 이는 '직무 적성'조차도 연봉을 결정짓는 절대적인 변수는 아님을 뜻합니다.")
        st.write("즉, 우리가 측정하고 있는 성적이나 시험 점수라는 **정량적 수치 전체가 실제 취업 시장의 복잡한 보상 체계를 온전히 설명하지 못한다**는 사실을 시사합니다.")
        
        st.divider()
        st.success("**🚀 최종 시사점**\n\n진로 교육은 획일화된 점수 경쟁을 넘어, 데이터로 측정되지 않는 **'비인지적 역량(실전 문제해결, 소통, 협업 등)'**을 강화하는 방향으로 패러다임이 전환되어야 합니다.")

else:
    st.warning("왼쪽 사이드바에서 파일을 업로드해 주십시오.")
