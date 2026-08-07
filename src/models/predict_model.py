import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


logger = logging.getLogger(__name__)


def load_model(
    model_path: str | Path = "models/RealEstateModel.pkl",
):
    """
    Load a trained real estate model from disk.
    """

    try:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file was not found at: {model_path}"
            )

        with model_path.open("rb") as model_file:
            model = pickle.load(model_file)

        logger.info("Model loaded from %s", model_path)

        return model

    except Exception as exc:
        logger.exception("Model loading failed.")

        raise RuntimeError(
            f"Unable to load the trained model: {exc}"
        ) from exc


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the regression model using MAE, RMSE and R².
    """

    try:
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        rmse = mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5

        r2 = r2_score(y_test, predictions)

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

        logger.info("Model evaluation results: %s", metrics)

        return metrics

    except Exception as exc:
        logger.exception("Model evaluation failed.")

        raise RuntimeError(
            f"Unable to evaluate the model: {exc}"
        ) from exc


def predict_property_price(
    model,
    input_data: pd.DataFrame,
) -> float:
    """
    Predict the price of one property.
    """

    try:
        prediction = model.predict(input_data)

        return float(prediction[0])

    except Exception as exc:
        logger.exception("Property-price prediction failed.")

        raise RuntimeError(
            f"Unable to predict the property price: {exc}"
        ) from exc