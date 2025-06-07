import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data safely with fallback
@st.cache_data
def load_data():
    try:
        train = pd.read_csv("lendingclub_traindata.csv")
        test = pd.read_csv("lendingclub_testdata.csv")
    except:
        st.error("CSV file loading failed. Please check file format and location.")
        return None, None

    expected_cols = ["home_ownership", "income", "dti", "fico", "loan_status"]
    if train.shape[1] == 5:
        train.columns = expected_cols
    if test.shape[1] == 5:
        test.columns = expected_cols

    train["income"] = train["income"] / 1000
    test["income"] = test["income"] / 1000

    return train, test

# Train logistic regression model
@st.cache_resource
def train_model(train):
    X_train = train[["home_ownership", "income", "dti", "fico"]]
    y_train = train["loan_status"]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Plot ROC curve at current threshold and highlight AUC
def plot_roc_with_point(y_true, y_scores, threshold):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')  # AUC 영역 강조
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')

    idx = np.argmin(np.abs(thresholds - threshold))
    ax.plot(fpr[idx], tpr[idx], 'o', color='green', markersize=10, label=f'Threshold = {threshold:.2f}')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve with AUC')
    ax.legend(loc='lower right')
    return fig, roc_auc

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'], 
                yticklabels=['No Default', 'Default'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

# Streamlit App
st.title("Loan Default Prediction - ROC Dashboard")

train, test = load_data()
if train is not None and test is not None:
    model = train_model(train)

    X_test = test[["home_ownership", "income", "dti", "fico"]]
    y_test = test["loan_status"]
    pred_probs = model.predict_proba(X_test)[:, 1]

    threshold = st.slider("Select Classification Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    pred_labels = (pred_probs >= threshold).astype(int)

    st.subheader("ROC Curve and AUC")
    roc_fig, roc_auc = plot_roc_with_point(y_test, pred_probs, threshold)
    st.pyplot(roc_fig)
    st.write(f"AUC Score: {roc_auc:.3f}")

    st.subheader(f"Confusion Matrix at Threshold = {threshold:.2f}")
    cm_fig = plot_confusion_matrix(y_test, pred_labels)
    st.pyplot(cm_fig)

    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
else:
    st.warning("Please make sure CSV files are correctly uploaded and have 5 columns.")
