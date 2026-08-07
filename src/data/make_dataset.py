import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load and perform basic preprocessing on the real estate dataset.

    Parameters
    ----------
    data_path : str or Path
        Location of the raw CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned real estate dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If the dataset is empty or does not contain the target column.
    RuntimeError
        If another data-loading error occurs.
    """

    try:
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset was not found at: {data_path}"
            )

        logger.info("Loading dataset from %s", data_path)

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("The real estate dataset is empty.")

        if "price" not in df.columns:
            raise ValueError(
                "The required target column 'price' is missing."
            )

        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            logger.info(
                "Removing %s duplicate rows.",
                duplicate_count
            )
            df = df.drop_duplicates()

        numeric_columns = df.select_dtypes(include="number").columns

        for column in numeric_columns:
            if df[column].isna().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)

        logger.info(
            "Dataset loaded successfully with shape %s",
            df.shape
        )

        return df

    except (FileNotFoundError, ValueError):
        logger.exception("Dataset validation failed.")
        raise

    except Exception as exc:
        logger.exception("Unexpected error while loading the dataset.")

        raise RuntimeError(
            f"Unable to load and preprocess the dataset: {exc}"
        ) from exc