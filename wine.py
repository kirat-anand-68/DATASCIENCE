import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="wide")


# ---------------------------------------------------------
# PREMIUM UI CSS
# ---------------------------------------------------------
premium_css = """
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px;
}

.gradient-title {
    font-size: 45px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff00cc, #3333ff);
    -webkit-background-clip: text;
    color: transparent;
    text-align: center;
    padding-bottom: 15px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e, #16213e);
    padding: 25px !important;
    box-shadow: 4px 0px 20px rgba(0,0,0,0.4);
}

.sidebar-title {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #ff00cc, #00ffff);
    -webkit-background-clip: text;
    color: transparent;
    padding-bottom: 10px;
}

div[role="radiogroup"] > label > div {
    font-size: 22px !important;
    font-weight: 600 !important;
    padding: 10px 12px;
    margin: 6px 0px;
    border-radius: 12px;
    transition: 0.3s;
}

div[role="radiogroup"] > label:hover {
    background-color: rgba(255,255,255,0.07);
    transform: scale(1.05);
}

div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, #ff00cc55, #00ffff33);
    box-shadow: 0px 0px 12px #7700ff;
    border-radius: 12px;
}

.author-card {
    margin-top: 40px;
    padding: 18px;
    border-radius: 15px;
    color: white;
    background: rgba(255,255,255,0.1);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0px 0px 10px rgba(255,255,255,0.12);
}

.glass-card {
    background: rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0px 0px 20px rgba(255,255,255,0.15);
    animation: floatCard 6s infinite ease-in-out;
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-7px); }
    100% { transform: translateY(0px); }
}

.pred-badge {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    padding: 25px;
    border-radius: 18px;
    margin-top: 25px;
    color: white;
    letter-spacing: 1px;
    animation: glow 1.5s infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0px 0px 10px rgba(255,255,255,0.3); }
    to   { box-shadow: 0px 0px 25px rgba(255,255,255,0.6); }
}

</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)



# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv('/Users/Kirat/OneDrive/Desktop/wine_quality_medium.csv')

# Remove outliers
z = np.abs(stats.zscore(df))
df_clean = df[(z < 3).all(axis=1)]

X = df_clean.drop("quality", axis=1)
y = df_clean["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))



# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("<h1 class='sidebar-title'>📌 Navigation</h1>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Home", "📊 Dataset Info", "🔮 Predict Quality"]
)

st.sidebar.markdown(
    """
    <div class='author-card'>
        <b>Developed by</b><br>
        Kirat Anand <br>
        <span style="font-size:15px;">Machine Learning + Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True
)



# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "🏠 Home":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:white;'>🍷 Wine Quality Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.write("This app predicts wine quality using Random Forest and a beautiful UI.")
    st.metric("✅ Model Accuracy", f"{accuracy*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------------------------------------
# DATASET INFO
# ---------------------------------------------------------
elif page == "📊 Dataset Info":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("📊 Dataset Information")
    st.write("### Dataset Preview:")
    st.dataframe(df_clean.head())
    st.write("### Shape:")
    st.write(df_clean.shape)
    st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif page == "🔮 Predict Quality":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("🔮 Predict Wine Quality")

    col1, col2, col3 = st.columns(3)

    inputs = []
    cols = [col1, col2, col3]

    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(col, float(df[col].min()), float(df[col].max()))
            inputs.append(val)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✅ Predict Now", use_container_width=True):
        input_array = np.array(inputs).reshape(1, -1)
        pred = model.predict(input_array)[0]

        if pred <= 4:
            color = "#ff4d4d"
        elif pred <= 6:
            color = "#ffa133"
        else:
            color = "#37c95c"

        st.markdown(
            f"<div class='pred-badge' style='background:{color};'>Predicted Quality: {pred}</div>",
            unsafe_allow_html=True
        )
