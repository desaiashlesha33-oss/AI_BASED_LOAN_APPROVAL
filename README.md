# AI-Based Loan Approval Prediction & Risk Analytics

## Internship Project
Data Analytics with AI — AICTE Internship

## Problem Statement
Financial institutions receive many loan applications and need to understand historical application patterns, identify risk drivers, and support consistent loan-approval analysis.

## Objectives
1. Clean and analyze loan application data.
2. Build an executive-style KPI view.
3. Identify trends and potential risk drivers.
4. Train a machine-learning classification model.
5. Provide an interactive prediction interface.
6. Convert analytical findings into actionable business insights.

## Dataset
This project uses a **synthetic educational dataset generated for this project** with 5,000 loan applications. It contains applicant profile, income, loan, credit-score, asset and loan-status fields.

Because the internship instructions require a dataset source/link in the README, the dataset file included in this project should be uploaded to the same GitHub repository. After upload, replace the placeholder below with the repository's raw CSV link:

`DATASET_LINK_TO_REPLACE_AFTER_GITHUB_UPLOAD`

The project deliberately does not claim that the generated dataset is an external real-world dataset.

## Main KPIs
- Total applications
- Approval rate
- Average applicant income
- Average CIBIL score
- Loan amount
- Loan-to-income context
- Approval/rejection distribution

## Analytics Flow
Data → Cleaning → EDA → KPIs → Trends → Drivers → Risks/Opportunities → ML prediction → Recommended actions

## Machine Learning
Random Forest Classifier is used for binary classification:
- Approved = 1
- Rejected = 0

The data is split into 80% training and 20% testing using a fixed random state.

## Evaluation on the project dataset
- Accuracy: 70.50%
- Precision: 75.04%
- Recall: 78.32%
- F1 Score: 76.64%

These metrics are specific to this generated dataset and should not be interpreted as real-world lending performance.

## Files
- `loan_approval_project.py` — single-file dashboard, model training and prediction interface
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `Loan_Approval_Project_Report.docx` — project report
- `loan_approval_dataset.csv` — dataset used by the application

## How to Run
```bash
pip install -r requirements.txt
streamlit run loan_approval_project.py
```

## Business Interpretation
The project is designed as a decision-support demonstration. Analysts can use the dashboard to understand application patterns, identify higher-risk segments, and investigate factors associated with approval outcomes. It should not be used as the sole basis for real lending decisions.

## Academic Note
This project was created as an educational internship project. Dataset generation, preprocessing, model training and evaluation are documented for reproducibility.
