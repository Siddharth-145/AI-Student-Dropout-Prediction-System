import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =============================
# CUSTOM CSS (Professional UI)
# =============================
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fb; }
    h1, h2, h3 { color: #102a43; }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .low { color: #2e7d32; font-weight: bold; }
    .medium { color: #ed6c02; font-weight: bold; }
    .high { color: #d32f2f; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# DATASET (Synthetic but realistic)
# =============================
data = {
    'attendance': [65, 70, 80, 90, 85, 60, 75, 88, 92, 55, 78, 83, 68, 74, 95, 50, 58, 62, 72, 81],
    'gpa': [5.5, 6.0, 7.2, 8.5, 8.0, 5.0, 6.8, 8.2, 9.0, 4.5, 7.0, 7.5, 6.0, 6.5, 9.2, 4.0, 5.0, 5.8, 6.7, 7.8],
    'internal_marks': [45, 50, 65, 80, 75, 40, 60, 78, 85, 35, 62, 70, 48, 58, 90, 30, 42, 50, 60, 72],
    'study_hours': [5, 6, 8, 12, 10, 4, 7, 11, 13, 3, 8, 9, 6, 7, 14, 2, 4, 5, 7, 9],
    'stress_level': [7, 6, 4, 2, 3, 8, 5, 3, 2, 9, 4, 4, 6, 5, 2, 9, 8, 7, 5, 3],
    'financial_issue': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    'dropout': [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df.drop('dropout', axis=1)
y = df['dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# =============================
# HEADER
# =============================
st.title("🎓 AI-Based Student Dropout Prediction & Retention System")
st.markdown("<p style='font-size:17px;'>A professional AI dashboard that predicts dropout risk, explains the reasons using Explainable AI, and provides actionable feedback.</p>", unsafe_allow_html=True)

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("📘 Student Profile")
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
gpa = st.sidebar.slider("Current GPA", 0.0, 10.0, 7.0)
internal_marks = st.sidebar.slider("Internal Marks", 0, 100, 60)
study_hours = st.sidebar.slider("Study Hours / Week", 0, 20, 7)
stress_level = st.sidebar.slider("Stress Level (1–10)", 1, 10, 5)
financial_issue = st.sidebar.selectbox("Financial Issues", ["No", "Yes"])
fin_val = 1 if financial_issue == "Yes" else 0

# =============================
# MAIN LOGIC
# =============================
if st.sidebar.button("🚀 Run AI Analysis"):

    input_data = np.array([[attendance, gpa, internal_marks, study_hours, stress_level, fin_val]])
    probability = model.predict_proba(input_data)[0][1] * 100

    if probability < 30:
        risk = "LOW"
        cls = "low"
        feedback = "Student is academically stable. Maintain consistency and regular revision."
    elif probability < 60:
        risk = "MEDIUM"
        cls = "medium"
        feedback = "Student shows warning signs. Early academic intervention is recommended."
    else:
        risk = "HIGH"
        cls = "high"
        feedback = "Student is at high risk of dropout. Immediate academic and counseling support required."

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Summary")
        st.metric("Dropout Probability", f"{probability:.2f}%")
        st.markdown(f"Risk Level: <span class='{cls}'>{risk}</span>", unsafe_allow_html=True)
        st.markdown(f"**AI Feedback:** {feedback}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍 Explainable AI – Feature Contribution")
        importances = model.feature_importances_
        features = X.columns
        top = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots()
        ax.barh([f[0].replace('_',' ').title() for f in top], [f[1]*100 for f in top])
        ax.set_xlabel("Contribution (%)")
        ax.invert_yaxis()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Personalized Intervention & Feedback")
    suggestions = []

    if attendance < 75:
        suggestions.append("📌 Improve attendance to at least 80%.")
    if internal_marks < 60:
        suggestions.append("📘 Attend remedial classes and strengthen fundamentals.")
    if study_hours < 7:
        suggestions.append("⏰ Increase focused study time by 1–2 hours daily.")
    if stress_level > 6:
        suggestions.append("🧘 Seek counseling or stress-management programs.")
    if fin_val == 1:
        suggestions.append("💰 Explore scholarships or financial assistance options.")

    if not suggestions:
        suggestions.append("✅ No major risk factors detected. Continue current performance.")

    for s in suggestions:
        st.write(s)

    st.markdown("</div>", unsafe_allow_html=True)

    st.success("AI analysis completed successfully.")
