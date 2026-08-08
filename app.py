import logging

import streamlit as st

from src.models.predict_model import (
    load_prediction_files,
    prepare_user_input,
    predict_loan,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - "
        "%(message)s"
    ),
)


logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💳",
    layout="centered",
)


st.title(
    "💳 Loan Eligibility Predictor"
)

st.write(
    """
    Enter the applicant's information below
    to predict loan eligibility.
    """
)


@st.cache_resource
def load_files():

    return load_prediction_files()


try:

    model, scaler, model_columns = (
        load_files()
    )

except Exception as exc:

    st.error(
        "The trained model could not be loaded. "
        "Run `python main.py` first."
    )

    logger.exception(
        "Model loading failed."
    )

    st.stop()


with st.form(
    "loan_form"
):

    st.subheader(
        "Applicant Information"
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
        ],
    )

    married = st.selectbox(
        "Married",
        [
            "No",
            "Yes",
        ],
    )

    dependents = st.selectbox(
        "Dependents",
        [
            "0",
            "1",
            "2",
            "3+",
        ],
    )

    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "Not Graduate",
        ],
    )

    self_employed = st.selectbox(
        "Self Employed",
        [
            "No",
            "Yes",
        ],
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=500.0,
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0,
    )

    loan_term = st.selectbox(
        "Loan Term",
        [
            360.0,
            180.0,
            120.0,
            84.0,
            60.0,
            36.0,
            12.0,
        ],
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[
            1.0,
            0.0,
        ],
        format_func=lambda x:
            "Good Credit History"
            if x == 1.0
            else "Poor Credit History",
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural",
        ],
    )

    submitted = (
        st.form_submit_button(
            "Check Loan Eligibility"
        )
    )


if submitted:

    try:

        user_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed":
                self_employed,
            "ApplicantIncome":
                applicant_income,
            "CoapplicantIncome":
                coapplicant_income,
            "LoanAmount":
                loan_amount,
            "Loan_Amount_Term":
                loan_term,
            "Credit_History":
                credit_history,
            "Property_Area":
                property_area,
        }

        input_df = prepare_user_input(
            user_data,
            model_columns,
        )

        prediction, probability = (
            predict_loan(
                model,
                scaler,
                input_df,
            )
        )

        st.divider()

        if prediction == 1:

            st.success(
                "✅ Loan Application "
                "Likely Approved"
            )

        else:

            st.error(
                "❌ Loan Application "
                "Likely Not Approved"
            )

        if probability is not None:

            st.metric(
                "Approval Probability",
                f"{probability:.1%}"
            )

    except Exception as exc:

        logger.exception(
            "Streamlit prediction failed."
        )

        st.error(
            f"Prediction could not be completed: {exc}"
        )


# ------------------------------------
# Visualizations
# ------------------------------------

st.divider()

st.subheader(
    "📊 Model Visualizations"
)


st.markdown(
    "### Loan Approval Distribution"
)

try:

    st.image(
        "loan_distribution.png",
        use_container_width=True,
    )

except Exception:

    st.info(
        "Run `python main.py` "
        "to generate this chart."
    )


st.markdown(
    "### Model Accuracy Comparison"
)

try:

    st.image(
        "model_comparison.png",
        use_container_width=True,
    )

except Exception:

    st.info(
        "Run `python main.py` "
        "to generate this chart."
    )