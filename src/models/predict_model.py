import logging
import pickle
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_prediction_files(
    model_path="models/LoanEligibilityModel.pkl",
    scaler_path="models/scaler.pkl",
    columns_path="models/model_columns.pkl",
):
    """
    Load trained model, scaler and training columns.
    """

    try:
        paths = [
            model_path,
            scaler_path,
            columns_path,
        ]

        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Required file missing: {path}"
                )

        with open(
            model_path,
            "rb"
        ) as file:
            model = pickle.load(file)

        with open(
            scaler_path,
            "rb"
        ) as file:
            scaler = pickle.load(file)

        with open(
            columns_path,
            "rb"
        ) as file:
            model_columns = pickle.load(file)

        return (
            model,
            scaler,
            model_columns
        )

    except Exception as exc:
        logger.exception(
            "Prediction files could not be loaded."
        )

        raise RuntimeError(
            f"Unable to load prediction files: {exc}"
        ) from exc


def prepare_user_input(
    user_data,
    model_columns,
):
    """
    Convert Streamlit user input into the same
    format used during training.
    """

    try:
        df = pd.DataFrame(
            [user_data]
        )

        categorical_columns = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            dtype=int
        )

        # Add missing dummy columns
        for column in model_columns:
            if column not in df.columns:
                df[column] = 0

        # Keep only training columns and correct order
        df = df[model_columns]

        return df

    except Exception as exc:
        logger.exception(
            "User input preparation failed."
        )

        raise RuntimeError(
            f"Unable to prepare user input: {exc}"
        ) from exc


def predict_loan(
    model,
    scaler,
    input_df,
):
    """
    Predict whether loan will be approved.
    """

    try:
        scaled_input = scaler.transform(
            input_df
        )

        prediction = model.predict(
            scaled_input
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):
            probability = model.predict_proba(
                scaled_input
            )[0][1]

        else:
            probability = None

        return (
            int(prediction),
            probability
        )

    except Exception as exc:
        logger.exception(
            "Loan prediction failed."
        )

        raise RuntimeError(
            f"Prediction failed: {exc}"
        ) from exc