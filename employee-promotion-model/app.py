"""Employee Promotion - decision-support dashboard.  Run: streamlit run employee-promotion-model/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

from hr_analytics import ui  # noqa: E402
from hr_analytics.config import TASKS  # noqa: E402

task = TASKS["promotion"]
st.set_page_config(page_title="Employee Promotion", layout="wide")
st.title("Employee Promotion - Decision Support")
ui.banner("Promotion models can encode historical bias; review the fairness diagnostic before "
          "using any ranking.")

df = ui.load_dataset(task)
page = st.sidebar.radio("Page", ["Dataset", "Exploratory analysis", "Model comparison",
                                 "Promotion review shortlist"])

if page == "Dataset":
    ui.render_overview(df, task)

elif page == "Exploratory analysis":
    st.subheader("Exploratory analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="department", y="is_promoted", ax=ax, errorbar=None)
        ax.set_ylabel("Promotion rate"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="is_promoted", y="avg_training_score", ax=ax)
        st.pyplot(fig); plt.close(fig)
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="is_promoted", y="length_of_service", ax=ax)
        st.pyplot(fig); plt.close(fig)
    with c4:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="previous_year_rating", y="is_promoted", ax=ax, errorbar=None)
        ax.set_ylabel("Promotion rate")
        st.pyplot(fig); plt.close(fig)
    st.caption("`previous_year_rating` is missing for employees in their first year of service, so the "
               "missingness is informative. The pipeline imputes it and adds a missing-indicator "
               "instead of dropping rows (an earlier version dropped about 15% of the data).")

elif page == "Model comparison":
    res = ui.get_results(df, task.name)
    ui.render_model_comparison(res, task)
    ui.render_fairness(df, res, task)

else:
    res = ui.get_results(df, task.name)
    st.subheader("Promotion review shortlist")
    st.caption("Scores are out-of-fold: each employee was scored by a model that never saw them. "
               "The shortlist is a starting point for managers' review, not a decision.")
    dept = st.selectbox("Department", sorted(df["department"].dropna().unique()))
    seats = st.number_input("Shortlist size", 1, 50, 5)
    d = df.loc[res["oof"].index].assign(promotion_score=res["oof"])
    short = (d[d["department"] == dept].sort_values("promotion_score", ascending=False)
             .head(int(seats))[["employee_id", "department", "avg_training_score",
                                "previous_year_rating", "awards_won", "promotion_score"]])
    st.dataframe(short.style.format({"promotion_score": "{:.1%}"}), width="stretch")
    st.info("Check this shortlist against the fairness diagnostic and against factors the model "
            "cannot see (peer feedback, role openings, context).")
