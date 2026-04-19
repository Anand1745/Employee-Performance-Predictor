# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_data, preprocess, split_features_target
from src.model import load_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide"
)

# -------------------------------
# LOAD DATA + MODEL
# -------------------------------
@st.cache_data
def get_data():
    df = load_data()
    df = preprocess(df)
    return df

@st.cache_resource
def get_model():
    return load_model()

df = get_data()
model = get_model()

X, y = split_features_target(df)

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Employee Performance Predictor Dashboard")

# -------------------------------
# KPI SECTION
# -------------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", len(df))

with col2:
    st.metric("Avg Salary", int(df["Salary"].mean()))

with col3:
    high_pct = (df["Performance"] == "High").mean() * 100
    st.metric("High Performers (%)", f"{high_pct:.1f}%")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("🔧 Employee Input")

age = st.sidebar.slider("Age", 20, 60, 30)
experience = st.sidebar.slider("Experience (Years)", 1, 20, 5)
salary = st.sidebar.slider("Salary", 20000, 150000, 50000)
training = st.sidebar.slider("Training Hours", 5, 100, 20)
department = st.sidebar.selectbox("Department", ["HR", "IT", "Sales"])

# -------------------------------
# INPUT DATA
# -------------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Experience": [experience],
    "Salary": [salary],
    "Training_Hours": [training],
    "Department_IT": [1 if department == "IT" else 0],
    "Department_Sales": [1 if department == "Sales" else 0]
})

input_data = input_data.reindex(columns=X.columns, fill_value=0)

# -------------------------------
# SELECTED PROFILE
# -------------------------------
st.subheader("📄 Selected Employee Profile")

st.json({
    "Age": age,
    "Experience": experience,
    "Salary": salary,
    "Training Hours": training,
    "Department": department
})

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.sidebar.button("Predict Performance"):

    prediction = model.predict(input_data)[0]

    st.subheader("🎯 Prediction Result")

    if prediction == "High":
        st.success("High Performer 🚀")
    elif prediction == "Medium":
        st.warning("Medium Performer ⚖️")
    else:
        st.error("Low Performer ⚠️")

    try:
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba)
        st.write(f"Confidence Score: **{confidence:.2f}**")
    except:
        pass

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# VISUAL STYLE
# -------------------------------
sns.set_style("darkgrid")
sns.set_palette("coolwarm")

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("📊 Insights & Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x="Performance", data=df, ax=ax)
    ax.set_title("Performance Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="Experience",
        y="Salary",
        hue="Performance",
        data=df,
        ax=ax
    )
    ax.set_title("Experience vs Salary")
    st.pyplot(fig)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("📈 Feature Importance")

importance = model.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

# -------------------------------
# ADVANCED SIMULATION (FINAL)
# -------------------------------
st.subheader("🔮 Performance Simulation (Dynamic Impact)")

sim_df = pd.DataFrame({
    "Experience": list(range(1, 21)),
    "Salary": [salary + i * 3000 for i in range(20)],
    "Training_Hours": [training + i * 2 for i in range(20)],
    "Age": [age] * 20,
    "Department_IT": [1 if department == "IT" else 0] * 20,
    "Department_Sales": [1 if department == "Sales" else 0] * 20
})

sim_df = sim_df.reindex(columns=X.columns, fill_value=0)

preds = model.predict(sim_df)

score_map = {"Low": 1, "Medium": 2, "High": 3}
sim_df["Score"] = [score_map[p] for p in preds]
sim_df["Performance_Label"] = preds

st.line_chart(sim_df.set_index("Experience")["Score"])

st.dataframe(
    sim_df[["Experience", "Salary", "Training_Hours", "Performance_Label"]],
    use_container_width=True
)

# -------------------------------
# WHAT-IF INSIGHTS
# -------------------------------
st.subheader("📊 What-if Insight")

if training < 20:
    st.warning("⚠️ Low training hours reduce performance")
elif training > 50:
    st.success("🚀 High training significantly boosts performance")

if salary < 40000:
    st.warning("💰 Low salary may impact performance")
elif salary > 100000:
    st.success("💰 High salary correlates with high performance")

if experience > 10:
    st.success("📈 Experienced employees perform better")

# -------------------------------
# EXPLANATION
# -------------------------------
st.subheader("🧠 How Prediction Works")

st.info("""
Performance increases with:
• More experience  
• Higher training hours  
• Better salary  

Use the sliders to explore real-time impact.
""")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("© 2024 Employee Performance Predictor | Built with Streamlit")