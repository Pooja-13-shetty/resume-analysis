# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

@st.cache_data
def load_data(path="data/processed/resumes_processed.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="data/processed/resume_classifier_lr.joblib"):
    return joblib.load(path)

df = load_data()
st.title("Resume Analysis Dashboard")

# KPIs
st.metric("Total resumes", len(df))
st.metric("Average years experience", round(df['years_experience'].dropna().mean(),2))

# Filter
min_years = st.slider("Min years experience", 0, 30, 0)
filtered = df[df['years_experience'].fillna(0) >= min_years]

# Show top skills
all_skills = pd.Series([skill for skills in filtered['skills_list'].dropna() for skill in eval(str(skills))]).value_counts().head(20)
st.bar_chart(all_skills)

# Run model on a sample resume
st.header("Predict on custom resume text")
text = st.text_area("Paste resume text here")
if st.button("Predict"):
    model = load_model()
    pred = model.predict([text])[0]
    st.write("Predicted label:", pred)
