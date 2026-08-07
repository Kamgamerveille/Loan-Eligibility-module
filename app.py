import logging

import pandas as pd
import streamlit as st

from src.features.build_features import FEATURE_COLUMNS
from src.models.predict_model import (
    load_model,
    predict_property_price,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    ),
)

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)


st.title("Real Estate Price Predictor")

st.write(
    """
    This application estimates the selling price of a property
    using a trained Random Forest regression model.
    """
)


@st.cache_resource
def get_model():
    """
    Load and cache the trained machine-learning model.
    """

    return load_model("models/RealEstateModel.pkl")


try:
    model = get_model()

except Exception as exc:
    st.error(
        "The trained model could not be loaded. "
        "Run `python main.py` before starting the application."
    )

    logger.exception("Streamlit could not load the model.")

    st.stop()


with st.form("real_estate_inputs"):
    st.subheader("Property Details")

    column1, column2, column3 = st.columns(3)

    with column1:
        year_sold = st.number_input(
            "Year Sold",
            min_value=1900,
            max_value=2100,
            value=2013,
            step=1,
        )

        property_tax = st.number_input(
            "Property Tax",
            min_value=0.0,
            value=250.0,
            step=10.0,
        )

        insurance = st.number_input(
            "Insurance",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )

        beds = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            value=3,
            step=1,
        )

        baths = st.number_input(
            "Number of Bathrooms",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )

    with column2:
        sqft = st.number_input(
            "Square Footage",
            min_value=1.0,
            value=1500.0,
            step=50.0,
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2100,
            value=2000,
            step=1,
        )

        lot_size = st.number_input(
            "Lot Size",
            min_value=0.0,
            value=5000.0,
            step=100.0,
        )

        basement = st.selectbox(
            "Basement",
            options=["No", "Yes"],
        )

    with column3:
        popular = st.selectbox(
            "Popular Area",
            options=["No", "Yes"],
        )

        recession = st.selectbox(
            "Sold During Recession",
            options=["No", "Yes"],
        )

        property_type = st.selectbox(
            "Property Type",
            options=["Non-Condo", "Condo"],
        )

    property_age = max(
        int(year_sold) - int(year_built),
        0,
    )

    st.info(
        f"Calculated property age: {property_age} years"
    )

    submitted = st.form_submit_button(
        "Predict Property Price"
    )


if submitted:
    try:
        input_values = {
            "year_sold": year_sold,
            "property_tax": property_tax,
            "insurance": insurance,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "year_built": year_built,
            "lot_size": lot_size,
            "basement": 1 if basement == "Yes" else 0,
            "popular": 1 if popular == "Yes" else 0,
            "recession": 1 if recession == "Yes" else 0,
            "property_age": property_age,
            "property_type_Condo": (
                1 if property_type == "Condo" else 0
            ),
        }

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURE_COLUMNS,
        )

        estimated_price = predict_property_price(
            model,
            input_df,
        )

        st.subheader("Prediction Result")

        st.success(
            f"Estimated property price: "
            f"${estimated_price:,.2f}"
        )

    except Exception as exc:
        logger.exception(
            "Prediction failed in the Streamlit application."
        )

        st.error(
            f"The prediction could not be completed: {exc}"
        )


st.divider()

st.subheader("Feature Importance")

try:
    st.image(
        "feature_importance.png",
        caption="Random Forest Feature Importance",
    )

except Exception:
    st.info(
        "Run `python main.py` to generate the "
        "feature-importance image."
    )


st.subheader("Actual vs Predicted Prices")

try:
    st.image(
        "actual_vs_predicted.png",
        caption="Actual and Predicted Property Prices",
    )

except Exception:
    st.info(
        "Run `python main.py` to generate the "
        "actual-versus-predicted image."
    )