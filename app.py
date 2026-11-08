
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. Application Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shopper Purchase Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("Online Shopper Purchase Analysis")

st.write(
    "This application evaluates five machine learning "
    "classification models using online shopping session data."
)

st.markdown("---")


# --------------------------------------------------
# 2. Define Model and Feature Information
# --------------------------------------------------

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]


# --------------------------------------------------
# 3. Load Saved Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


try:
    models = load_models()
    st.success("All five trained models loaded successfully.")

except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.stop()


# --------------------------------------------------
# 4. Dataset Upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        st.info("Using the uploaded test dataset.")

    except Exception as error:
        st.error(f"Unable to read CSV file: {error}")
        st.stop()

else:

    default_file = "test_data.csv"

    if not os.path.exists(default_file):
        st.warning(
            "Please upload a test CSV file to continue."
        )
        st.stop()

    data = pd.read_csv(default_file)

    st.info(
        "Using the sample test dataset included in the repository. "
        "You can upload another CSV to evaluate it."
    )


# --------------------------------------------------
# 5. Validate Dataset
# --------------------------------------------------

required_columns = FEATURE_COLUMNS + ["Revenue"]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {missing_columns}"
    )
    st.stop()

if data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if data[required_columns].isnull().any().any():
    st.error(
        "The dataset contains missing values. "
        "Please provide a cleaned test dataset."
    )
    st.stop()

# Prepare input features
X_test = data[FEATURE_COLUMNS].copy()

# Convert target variable into binary values
if pd.api.types.is_bool_dtype(data["Revenue"]):

    y_test = data["Revenue"].astype(int)

else:

    numeric_target = pd.to_numeric(
        data["Revenue"],
        errors="coerce"
    )

    if numeric_target.notna().all() and set(
        numeric_target.unique()
    ).issubset({0, 1}):

        y_test = numeric_target.astype(int)

    else:

        target_mapping = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        }

        y_test = (
            data["Revenue"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

if y_test.isnull().any():
    st.error("Revenue must contain only True/False or 1/0 values.")
    st.stop()

y_test = y_test.astype(int)

if not set(y_test.unique()).issubset({0, 1}):
    st.error("Revenue must be a binary target with values 0 and 1.")
    st.stop()


# --------------------------------------------------
# 6. Display Dataset Information
# --------------------------------------------------

st.header("2. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sessions", f"{len(data):,}")
col2.metric("Input Features", len(FEATURE_COLUMNS))
col3.metric("Purchase Sessions", int(y_test.sum()))

st.write("Uploaded dataset preview:")
st.dataframe(data.head(), use_container_width=True)


# --------------------------------------------------
# 7. Evaluate All Five Models
# --------------------------------------------------

st.header("3. Model Performance Comparison")

evaluation_results = []
model_predictions = {}

for model_name, model in models.items():

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate score for AUC
    classifier = model.named_steps["classifier"]

    if hasattr(model, "predict_proba"):

        positive_index = list(
            classifier.classes_
        ).index(1)

        y_score = model.predict_proba(
            X_test
        )[:, positive_index]

    else:

        y_score = model.decision_function(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_score)
    else:
        auc = np.nan

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    model_predictions[model_name] = y_pred


results_df = pd.DataFrame(evaluation_results)

st.dataframe(
    results_df.set_index("Model").style.format("{:.4f}"),
    use_container_width=True
)

# Download comparison results
csv_results = results_df.to_csv(index=False)

st.download_button(
    label="Download Model Comparison CSV",
    data=csv_results,
    file_name="model_comparison.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 8. Select a Model
# --------------------------------------------------

st.header("4. Individual Model Evaluation")

selected_model = st.selectbox(
    "Select a classification model",
    list(models.keys()),
    index=list(models.keys()).index("Support Vector Machine")
)

selected_results = results_df[
    results_df["Model"] == selected_model
].iloc[0]

st.subheader(f"{selected_model} – Evaluation Metrics")

metric_columns = st.columns(3)

metric_columns[0].metric(
    "Accuracy",
    f"{selected_results['Accuracy']:.2%}"
)

metric_columns[1].metric(
    "AUC Score",
    f"{selected_results['AUC']:.4f}"
)

metric_columns[2].metric(
    "Precision",
    f"{selected_results['Precision']:.4f}"
)

metric_columns[0].metric(
    "Recall",
    f"{selected_results['Recall']:.4f}"
)

metric_columns[1].metric(
    "F1 Score",
    f"{selected_results['F1 Score']:.4f}"
)

metric_columns[2].metric(
    "MCC",
    f"{selected_results['MCC']:.4f}"
)


# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

st.header("5. Confusion Matrix and Classification Report")

selected_predictions = model_predictions[selected_model]

col1, col2 = st.columns(2)

with col1:

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        selected_predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Purchase", "Purchase"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    st.pyplot(fig)
    plt.close(fig)


with col2:

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        selected_predictions,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


# --------------------------------------------------
# 10. Actual and Predicted Distribution
# --------------------------------------------------

st.header("6. Actual vs Predicted Distribution")

distribution_df = pd.DataFrame({
    "Actual": y_test.value_counts(normalize=True),
    "Predicted": pd.Series(
        selected_predictions
    ).value_counts(normalize=True)
}).reindex([0, 1]).fillna(0)

distribution_df.index = [
    "No Purchase",
    "Purchase"
]

st.dataframe(
    (distribution_df * 100).round(2),
    use_container_width=True
)

st.caption(
    "All evaluation results are calculated using the "
    "uploaded test data and previously trained models. "
    "No model retraining is performed in this application."
)

st.markdown("---")

st.write("Machine Learning Assignment – Classification Models")
