import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Loan Approval Risk Analytics", layout="wide")
DATA_FILE = "loan_approval_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["loan_status", "loan_id"])
    y = df["loan_status"].map({"Rejected": 0, "Approved": 1})
    cat = ["education", "self_employed"]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)])
    model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred)}
    return pipe, results

df = load_data()
pipe, results = train_model(df)

st.title("AI-Based Loan Approval Prediction & Risk Analytics")
st.caption("Data Analytics + Machine Learning decision-support dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Applications", f"{len(df):,}")
c2.metric("Approval Rate", f"{df.loan_status.eq('Approved').mean()*100:.1f}%")
c3.metric("Average Income", f"₹{df.income_annum.mean()/1e6:.2f}M")
c4.metric("Average CIBIL", f"{df.cibil_score.mean():.0f}")

tab1, tab2, tab3 = st.tabs(["Executive Overview", "Risk & Drivers", "AI Prediction"])
with tab1:
    st.subheader("Business Overview")
    col1, col2 = st.columns(2)
    with col1: st.bar_chart(df["loan_status"].value_counts())
    with col2:
        edu = df.groupby("education")["loan_status"].apply(lambda s: (s == "Approved").mean()).rename("approval_rate")
        st.bar_chart(edu)
    st.markdown("### Key Insights")
    st.write("The dashboard focuses on approval rate, applicant financial profile, credit score, and loan characteristics so users can identify trends and potential risk drivers.")

with tab2:
    st.subheader("Risk and Opportunity Analysis")
    col1, col2 = st.columns(2)
    with col1:
        cibil_band = pd.cut(df["cibil_score"], bins=[299,500,600,700,800,900], labels=["300-500","501-600","601-700","701-800","801-900"])
        risk = df.assign(cibil_band=cibil_band).groupby("cibil_band", observed=False)["loan_status"].apply(lambda s: (s == "Rejected").mean())
        st.bar_chart(risk)
        st.caption("Higher rejection concentration can indicate a risk segment.")
    with col2:
        d = df.copy(); d["loan_to_income"] = d["loan_amount"] / d["income_annum"].replace(0, np.nan)
        st.bar_chart(d.groupby("education")["loan_to_income"].mean())
        st.caption("Loan-to-income ratio helps contextualize requested borrowing.")

with tab3:
    st.subheader("Applicant Prediction")
    left, right = st.columns(2)
    with left:
        dependents = st.number_input("Dependents", 0, 10, 2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        income = st.number_input("Annual Income (₹)", 200000, 50000000, 5000000, step=100000)
        loan_amount = st.number_input("Loan Amount (₹)", 100000, 50000000, 15000000, step=100000)
    with right:
        term = st.selectbox("Loan Term (years)", [2,4,6,8,10,12,14,16,18,20], index=5)
        cibil = st.slider("CIBIL Score", 300, 900, 700)
        residential = st.number_input("Residential Assets (₹)", -500000, 50000000, 8000000, step=100000)
        commercial = st.number_input("Commercial Assets (₹)", 0, 50000000, 5000000, step=100000)
        luxury = st.number_input("Luxury Assets (₹)", 0, 50000000, 10000000, step=100000)
        bank = st.number_input("Bank Assets (₹)", 0, 50000000, 5000000, step=100000)
    if st.button("Predict Loan Outcome"):
        sample = pd.DataFrame([{"no_of_dependents": dependents, "education": education, "self_employed": self_employed, "income_annum": income, "loan_amount": loan_amount, "loan_term": term, "cibil_score": cibil, "residential_assets_value": residential, "commercial_assets_value": commercial, "luxury_assets_value": luxury, "bank_asset_value": bank}])
        prediction = pipe.predict(sample)[0]
        probability = pipe.predict_proba(sample)[0][1]
        if prediction == 1: st.success(f"Model output: Approved (model probability: {probability:.1%})")
        else: st.warning(f"Model output: Rejected (model probability of approval: {probability:.1%})")

st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy: {results['accuracy']:.2%}")
st.sidebar.write(f"Precision: {results['precision']:.2%}")
st.sidebar.write(f"Recall: {results['recall']:.2%}")
st.sidebar.write(f"F1 Score: {results['f1']:.2%}")
st.sidebar.info("Educational decision-support project; not for real lending decisions.")
