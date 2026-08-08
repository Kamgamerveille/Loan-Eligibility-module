import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

TARGET_COLUMN = "Loan_Approved"


def build_features(
    df,
    output_path=(
        "data/processed/"
        "Processed_Credit_Dataset.csv"
    ),
):
    """
    Encode categorical variables and separate
    features from target.
    """

    try:
        df = df.copy()

        # Convert categorical variables to dummy variables
        df = pd.get_dummies(
            df,
            columns=CATEGORICAL_COLUMNS,
            dtype=int
        )

        # Convert target Y/N into 1/0
        df[TARGET_COLUMN] = df[
            TARGET_COLUMN
        ].replace(
            {
                "Y": 1,
                "N": 0,
            }
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            "Processed dataset saved to %s",
            output_path
        )

        X = df.drop(
            TARGET_COLUMN,
            axis=1
        )

        y = df[TARGET_COLUMN].astype(int)

        return X, y

    except Exception as exc:
        logger.exception(
            "Feature engineering failed."
        )

        raise RuntimeError(
            f"Unable to build features: {exc}"
        ) from exc