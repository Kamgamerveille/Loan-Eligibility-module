import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_and_clean_data(data_path):
    """
    Load and clean the raw loan eligibility dataset.
    """

    try:
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {data_path}"
            )

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("The dataset is empty.")

        logger.info(
            "Dataset loaded successfully with shape %s",
            df.shape
        )

        # Convert these columns to categorical/object
        df["Credit_History"] = df[
            "Credit_History"
        ].astype("object")

        df["Loan_Amount_Term"] = df[
            "Loan_Amount_Term"
        ].astype("object")

        # Fill missing categorical values
        df["Gender"] = df["Gender"].fillna(
            df["Gender"].mode()[0]
        )

        df["Married"] = df["Married"].fillna(
            df["Married"].mode()[0]
        )

        df["Dependents"] = df["Dependents"].fillna(
            df["Dependents"].mode()[0]
        )

        df["Self_Employed"] = df[
            "Self_Employed"
        ].fillna(
            df["Self_Employed"].mode()[0]
        )

        df["Loan_Amount_Term"] = df[
            "Loan_Amount_Term"
        ].fillna(
            df["Loan_Amount_Term"].mode()[0]
        )

        df["Credit_History"] = df[
            "Credit_History"
        ].fillna(
            df["Credit_History"].mode()[0]
        )

        # Fill numerical missing value
        df["LoanAmount"] = df[
            "LoanAmount"
        ].fillna(
            df["LoanAmount"].median()
        )

        # Drop ID because it is not useful for prediction
        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)

        logger.info(
            "Data cleaning completed. Remaining missing values: %s",
            df.isnull().sum().sum()
        )

        return df

    except Exception as exc:
        logger.exception("Data loading/cleaning failed.")

        raise RuntimeError(
            f"Unable to prepare dataset: {exc}"
        ) from exc