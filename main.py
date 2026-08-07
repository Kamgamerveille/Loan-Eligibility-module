import logging

from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.predict_model import evaluate_model
from src.models.train_model import train_real_estate_model
from src.visualization.visualize import (
    plot_actual_vs_predicted,
    plot_feature_importance,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    ),
)

logger = logging.getLogger(__name__)


def main():
    """
    Run the complete Real Estate machine-learning pipeline.
    """

    try:
        data_path = "data/raw/final.csv"

        df = load_and_preprocess_data(data_path)

        X, y = build_features(df)

        (
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_real_estate_model(X, y)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        predictions = model.predict(X_test)

        plot_feature_importance(model, X)

        plot_actual_vs_predicted(
            y_test,
            predictions,
        )

        print("\nReal Estate Model Results")
        print("-------------------------")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")
        print(f"MAE: ${metrics['mae']:,.2f}")
        print(f"RMSE: ${metrics['rmse']:,.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")

    except Exception as exc:
        logger.exception(
            "The Real Estate pipeline could not complete."
        )

        print(f"Pipeline error: {exc}")


if __name__ == "__main__":
    main()