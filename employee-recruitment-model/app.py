"""Candidate Hiring - decision-support dashboard.  Run: streamlit run employee-recruitment-model/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

from hr_analytics import ui  # noqa: E402
from hr_analytics.config import TASKS  # noqa: E402

task = TASKS["recruitment"]
st.set_page_config(page_title="Candidate Hiring", layout="wide")
st.title("Candidate Hiring - Decision Support")
ui.banner("Gender and age are not model inputs, but check the fairness diagnostic for proxies.")

df = ui.load_dataset(task)
page = st.sidebar.radio("Page", ["Dataset", "Exploratory analysis", "Model comparison",
                                 "Candidate shortlist"])

STRATEGIES = {  # numeric codes used in the dataset, as in the original app
    "Technical Hiring (Engineering / IT)": 1,
    "Corporate Hiring (HR / Admin)": 2,
    "Sales & Marketing Hiring": 3,
}

if page == "Dataset":
    ui.render_overview(df, task)

elif page == "Exploratory analysis":
    st.subheader("Exploratory analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="RecruitmentStrategy", y="HiringDecision", ax=ax, errorbar=None)
        ax.set_ylabel("Hiring rate")
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="HiringDecision", y="SkillScore", ax=ax)
        st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="HiringDecision", y="InterviewScore", ax=ax)
    st.pyplot(fig); plt.close(fig)

elif page == "Model comparison":
    res = ui.get_results(df, task.name)
    ui.render_model_comparison(res, task)
    ui.render_fairness(df, res, task)

else:
    res = ui.get_results(df, task.name)
    st.subheader("Candidate shortlist")
    st.caption("Candidates are filtered by your requirements, then ranked by out-of-fold model score "
               "(each candidate scored by a model that never saw them). Past hiring decisions in the "
               "file are the labels, so the model learns past human behaviour, including its biases.")
    label = st.selectbox("Hiring track", list(STRATEGIES))
    mexp = st.slider("Minimum years of experience", 0, int(df["ExperienceYears"].max()), 2)
    mint = st.slider("Minimum interview score", 0, 100, 40)
    msk = st.slider("Minimum skill score", 0, 100, 50)
    pool = df.loc[res["oof"].index].assign(model_score=res["oof"])
    pool = pool[(pool["RecruitmentStrategy"] == STRATEGIES[label]) & (pool["ExperienceYears"] >= mexp)
                & (pool["InterviewScore"] >= mint) & (pool["SkillScore"] >= msk)]
    if pool.empty:
        st.warning("No candidates match these requirements. Relax a filter.")
    else:
        n = st.slider("Shortlist size", 1, min(50, len(pool)), min(5, len(pool)))
        show = pool.sort_values("model_score", ascending=False).head(n).drop(
            columns=["DistanceFromCompany", "PersonalityScore", "HiringDecision"], errors="ignore")
        st.write(f"{len(pool)} candidates match.")
        st.dataframe(show.style.format({"model_score": "{:.1%}"}), width="stretch")
