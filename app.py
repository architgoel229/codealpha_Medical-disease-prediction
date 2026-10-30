
import os
import joblib
import streamlit as st
import pandas as pd

from sklearn.metrics import confusion_matrix

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .sub-title {
        color: #999999;
        margin-bottom: 25px;
    }

    .model-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 12px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODELS

MODEL_DIR = "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl"
}


@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models


try:
    models = load_models()

except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()


# MODEL PERFORMANCE

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "SVM",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [0.868852, 0.819672, 0.868852, 0.852459],
    "Precision": [0.812500, 0.757576, 0.833333, 0.806452],
    "Recall": [0.928571, 0.892857, 0.892857, 0.892857],
    "F1 Score": [0.866667, 0.819672, 0.862069, 0.847458],
    "ROC-AUC": [0.959957, 0.943723, 0.883117, 0.951299]
})


# HEADER

st.markdown(
    '<p class="main-title">❤️ Heart Disease Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">'
    'Machine Learning Based Classification | 4 Tuned Models'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# NAVIGATION

tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Model Comparison",
    "About Project"
])


# TAB 1: PREDICTION

with tab1:

    st.subheader("Enter Patient Information")

    st.write(
        "Provide the following measurements to compare "
        "predictions from all four models."
    )

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=50
            )

            sex = st.selectbox(
                "Sex",
                options=[0, 1],
                format_func=lambda x: {
                    0: "Female",
                    1: "Male"
                }[x]
            )

            cp = st.selectbox(
                "Chest Pain Type",
                options=[1, 2, 3, 4]
            )

            trestbps = st.number_input(
                "Resting Blood Pressure",
                min_value=50,
                max_value=250,
                value=120
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=50,
                max_value=600,
                value=200
            )

        with col2:

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            restecg = st.selectbox(
                "Resting ECG",
                options=[0, 1, 2]
            )

            thalach = st.number_input(
                "Maximum Heart Rate",
                min_value=50,
                max_value=250,
                value=150
            )

            exang = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:

            oldpeak = st.number_input(
                "Oldpeak",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

            slope = st.selectbox(
                "Slope of ST Segment",
                options=[1, 2, 3]
            )

            ca = st.selectbox(
                "Number of Major Vessels",
                options=[0, 1, 2, 3]
            )

            thal = st.selectbox(
                "Thal",
                options=[3, 6, 7]
            )

        submitted = st.form_submit_button(
            "Predict",
            width='stretch'
        )


    # PREDICTIONS

    if submitted:

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        st.divider()
        st.subheader("Prediction Results")

        results = []

        columns = st.columns(4)

        for (name, model), col in zip(models.items(), columns):

            prediction = int(model.predict(input_data)[0])

            probability = None

            if hasattr(model, "predict_proba"):

                probability = float(
                    model.predict_proba(input_data)[0][1]
                )

            label = (
                "Disease detected"
                if prediction == 1
                else "No disease detected"
            )

            with col:

                st.markdown(
                    f'<div class="model-card">'
                    f'<h4>{name}</h4>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if prediction == 1:
                    st.error(label)
                else:
                    st.success(label)

                if probability is not None:

                    st.metric(
                        "Predicted probability",
                        f"{probability:.1%}"
                    )

            results.append({
                "Model": name,
                "Prediction": label,
                "Probability": (
                    f"{probability:.2%}"
                    if probability is not None
                    else "Not available"
                )
            })

        st.subheader("Prediction Comparison")

        st.dataframe(
            pd.DataFrame(results),
            width='stretch',
            hide_index=True
        )

        positive_count = sum(
            model.predict(input_data)[0] == 1
            for model in models.values()
        )

        st.info(
            f"{positive_count} out of 4 models predicted "
            "the positive class."
        )

        st.warning(
            "Educational demonstration only. These predictions "
            "are not medical diagnoses. Do not use this application "
            "to make medical decisions."
        )


# TAB 2: MODEL COMPARISON

with tab2:

    st.subheader("Test Set Performance")

    st.write(
        "Performance of the four tuned models on the held-out "
        "test dataset."
    )

    st.dataframe(
        performance.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        width='stretch',
        hide_index=True
    )

    st.divider()

    st.subheader("Metric Comparison")

    metric = st.selectbox(
        "Select Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    chart_data = performance.set_index("Model")[[metric]]

    st.bar_chart(chart_data)

    st.divider()

    st.subheader("Confusion Matrices")

    confusion_data = {
        "Logistic Regression": [[27, 6], [2, 26]],
        "SVM": [[25, 8], [3, 25]],
        "Decision Tree": [[28, 5], [3, 25]],
        "Random Forest": [[27, 6], [3, 25]]
    }

    cols = st.columns(2)

    for i, (name, matrix) in enumerate(confusion_data.items()):

        with cols[i % 2]:

            st.markdown(f"**{name}**")

            cm_df = pd.DataFrame(
                matrix,
                index=["Actual: No Disease", "Actual: Disease"],
                columns=["Predicted: No Disease", "Predicted: Disease"]
            )

            st.dataframe(
                cm_df,
                width='stretch'
            )


# TAB 3: ABOUT PROJECT

with tab3:

    st.subheader("About This Project")

    st.write("""
    This project uses machine learning classification models
    to predict the presence or absence of heart disease
    based on patient attributes.

    **Dataset:** UCI Cleveland Heart Disease Dataset

    **Models implemented:**
    - Logistic Regression
    - Support Vector Machine
    - Decision Tree
    - Random Forest

    **Model optimization:**
    - Optuna hyperparameter optimization
    - 50 trials per model
    - 5-fold stratified cross-validation
    - F1 and recall-oriented optimization

    **Preprocessing:**
    - Missing value imputation
    - Feature transformation
    - Categorical feature encoding
    - SMOTE for selected models
    """)

    st.divider()

    st.subheader("Disclaimer")

    st.warning("""
    This project is intended for educational and research
    purposes only. It is not a clinically validated medical
    diagnostic system and must not replace professional
    medical advice, diagnosis, or treatment.
    """)


# FOOTER

st.divider()

st.caption(
    "Heart Disease Prediction | Machine Learning Project"
)
