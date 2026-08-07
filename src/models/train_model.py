import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def train_real_estate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: str | Path = "models/RealEstateModel.pkl",
):
    """
    Split the data, train a Random Forest regression model,
    and save the trained model.

    Parameters
    ----------
    X : pandas.DataFrame
        Model input features.
    y : pandas.Series
        Property-price target.
    model_path : str or Path
        Location where the trained model will be stored.

    Returns
    -------
    tuple
        Trained model, X_train, X_test, y_train and y_test.
    """

    try:
        if X.empty or y.empty:
            raise ValueError(
                "Features and target must not be empty."
            )

        stratify_values = None

        if "property_type_Condo" in X.columns:
            stratify_values = X["property_type_Condo"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify_values,
        )

        model = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=42,
            n_jobs=-1,
        )

        logger.info("Training Random Forest Regressor.")

        model.fit(X_train, y_train)

        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as model_file:
            pickle.dump(model, model_file)

        logger.info(
            "Trained model saved to %s",
            output_path
        )

        return model, X_train, X_test, y_train, y_test

    except Exception as exc:
        logger.exception("Model training failed.")

        raise RuntimeError(
            f"Unable to train the real estate model: {exc}"
        ) from exc