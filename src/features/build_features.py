import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


TARGET_COLUMN = "price"

FEATURE_COLUMNS = [
    "year_sold",
    "property_tax",
    "insurance",
    "beds",
    "baths",
    "sqft",
    "year_built",
    "lot_size",
    "basement",
    "popular",
    "recession",
    "property_age",
    "property_type_Condo",
]


def build_features(
    df: pd.DataFrame,
    processed_data_path: str | Path = (
        "data/processed/Processed_Real_Estate_Dataset.csv"
    ),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select model features and separate the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned real estate dataset.
    processed_data_path : str or Path
        Location where the processed dataset will be stored.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Input features and target variable.
    """

    try:
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "The following required columns are missing: "
                + ", ".join(missing_columns)
            )

        processed_df = df[required_columns].copy()

        output_path = Path(processed_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        processed_df.to_csv(output_path, index=False)

        logger.info(
            "Processed dataset saved to %s",
            output_path
        )

        X = processed_df[FEATURE_COLUMNS]
        y = processed_df[TARGET_COLUMN]

        logger.info(
            "Features created successfully. X shape: %s, y shape: %s",
            X.shape,
            y.shape,
        )

        return X, y

    except Exception as exc:
        logger.exception("Feature-building process failed.")

        raise RuntimeError(
            f"Unable to build model features: {exc}"
        ) from exc