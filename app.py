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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Real Estate Price Predictor")

st.write(
    "Enter the property details below to estimate the selling price."
)


@st.cache_resource
def get_model():
    return load_model("models/RealEstateModel.pkl")


try:
    model = get_model()

except Exception as exc:
    st.error("The model could not be loaded. Run `python main.py` first.")
    logger.exception("Model loading failed.")
    st.stop()


with st.form("property_form"):

    st.subheader("Property Information")

    beds = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    baths = st.number_input(
        "Bathrooms",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    sqft = st.number_input(
        "Square Footage",
        min_value=200.0,
        value=1500.0,
        step=100.0,
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2000,
        step=1,
    )

    property_type = st.selectbox(
        "Property Type",
        ["House / Other", "Condo"],
    )

    basement = st.selectbox(
        "Basement",
        ["No", "Yes"],
    )

    st.subheader("Additional Information")

    property_tax = st.number_input(
        "Property Tax",
        min_value=0.0,
        value=250.0,
        step=25.0,
    )

    insurance = st.number_input(
        "Insurance",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    submitted = st.form_submit_button("Predict Price")


if submitted:

    try:
        # Values automatically handled by the app
        year_sold = 2013
        lot_size = 5000.0
        popular = 0
        recession = 0

        property_age = max(
            year_sold - year_built,
            0,
        )

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
            "popular": popular,
            "recession": recession,
            "property_age": property_age,
            "property_type_Condo": (
                1 if property_type == "Condo" else 0
            ),
        }

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURE_COLUMNS,
        )

        prediction = predict_property_price(
            model,
            input_df,
        )

        st.success(
            f"Estimated Property Price: ${prediction:,.2f}"
        )

    except Exception as exc:
        logger.exception("Prediction failed.")
        st.error(f"Prediction failed: {exc}")