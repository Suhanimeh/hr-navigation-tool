"""Employee Attrition - decision-support dashboard.  Run: streamlit run employee-attrition-model/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

from hr_analytics import data as D  # noqa: E402
from hr_analytics import ui  # noqa: E402
from hr_analytics.config import TASKS  # noqa: E402

task = TASKS["attrition"]
st.set_page_config(page_title="Employee Attrition", layout="wide")
st.title("Employee Attrition - Decision Support")
ui.banner()

df = ui.load_dataset(task)
page = st.sidebar.radio("Page", ["Dataset", "Exploratory analysis", "Model comparison",
                                 "Attrition risk"])

if page == "Dataset":
    ui.render_overview(df, task)

elif page == "Exploratory analysis":
    st.subheader("Exploratory analysis")
    d = df.assign(left=(df[task.target] == task.positive_label))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x=d.Department, y=d.left * 100, ax=ax, errorbar=None)
        ax.set_ylabel("Attrition rate (%)"); ax.set_title("Attrition rate by department")
        plt.setp(ax.get_xticklabels(), rotation=20)
        st.pyplot(fig); plt.close(fig)
    with c2:
        if "OverTime" in d:
            fig, ax = plt.subplots()
            sns.barplot(x=d.OverTime, y=d.left * 100, ax=ax, errorbar=None)
            ax.set_ylabel("Attrition rate (%)"); ax.set_title("Attrition rate by overtime")
            st.pyplot(fig); plt.close(fig)
    num = st.selectbox("Numeric feature", df.select_dtypes("number").columns)
    fig, ax = plt.subplots()
    sns.boxplot(data=d, x=task.target, y=num, ax=ax)
    st.pyplot(fig); plt.close(fig)

elif page == "Model comparison":
    res = ui.get_results(df, task.name)
    ui.render_model_comparison(res, task)
    ui.render_fairness(df, res, task)

else:
    res = ui.get_results(df, task.name)
    base = res["y"].mean()
    st.subheader("Attrition risk for one employee profile")
    st.caption(f"Model: {res['best']}. The score is a probability that this profile resembles past "
               f"leavers in the training data (base rate {base:.1%}). It does not say *when* anyone "
               "might leave: the dataset has no time-to-event information.")
    X = res["X"]
    vals = {}
    cols = st.columns(3)
    for i, c in enumerate(X.columns):
        with cols[i % 3]:
            if not pd.api.types.is_numeric_dtype(X[c]):  # works for object and pandas>=3 str dtype
                vals[c] = st.selectbox(c, sorted(X[c].dropna().unique()))
            else:
                lo, hi = int(X[c].min()), int(X[c].max())
                vals[c] = st.slider(c, lo, hi, int(X[c].median()))
    if st.button("Estimate risk"):
        p = res["fitted"][res["best"]].predict_proba(pd.DataFrame([vals]))[0, 1]
        st.metric("Estimated attrition risk", f"{p:.1%}", f"{p / base:.1f}x the average rate")
        st.info("Use as a prompt for a supportive retention conversation, never for adverse action.")

    st.divider()
    st.subheader("Team-level view (aggregate, out-of-fold scores)")
    st.caption("Each employee is scored by a model that never saw them. Only group averages are "
               "shown on purpose: individual 'highest-risk' lists invite misuse.")
    group = st.selectbox("Group by", [c for c in ["Department", "JobRole", "OverTime"] if c in df.columns])
    agg = (pd.DataFrame({group: df.loc[res["oof"].index, group], "risk": res["oof"]})
           .groupby(group)["risk"].agg(["mean", "count"]).sort_values("mean", ascending=False))
    agg.columns = ["mean_predicted_risk", "employees"]
    st.dataframe(agg.style.format({"mean_predicted_risk": "{:.1%}"}))
