# Loan Eligibility Prediction

## Overview

This project predicts whether a loan application is likely
to be approved based on applicant financial and demographic
information.

The project was developed as part of CST2216 - Business
Intelligence System Infrastructure at Algonquin College.

## Machine Learning Models

Three classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model is automatically selected and saved.

## Dataset

The dataset contains information including:

- Gender
- Marital Status
- Dependents
- Education
- Employment Status
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Loan Approval Status

## Project Structure


loan_eligibility_module/
- app.py
- main.py
- requirements.txt
- runtime.txt
- README.md
- .gitignore
- data/
    - raw/
    - processed/
- models/
    - LoanEligibilityModel.pkl
- src/
    - data/
        - make_dataset.py
    - features/
        - build_features.py
    - models/
        - train_model.py
        - predict_model.py
    - visualization/
        - visualize.py