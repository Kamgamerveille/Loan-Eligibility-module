import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(
    y_test,
    predictions,
    output_path: str | Path = "actual_vs_predicted.png",
):
    """
    Create and save an actual-versus-predicted scatter plot.
    """

    try:
        output_path = Path(output_path)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(y_test, predictions, alpha=0.6)

        minimum = min(y_test.min(), predictions.min())
        maximum = max(y_test.max(), predictions.max())

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        ax.set_title("Actual vs Predicted Property Prices")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")

        fig.tight_layout()
        fig.savefig(output_path)

        plt.close(fig)

        logger.info(
            "Actual-versus-predicted plot saved to %s",
            output_path,
        )

    except Exception as exc:
        logger.exception(
            "Actual-versus-predicted visualization failed."
        )

        raise RuntimeError(
            f"Unable to create prediction visualization: {exc}"
        ) from exc


def plot_feature_importance(
    model,
    X: pd.DataFrame,
    output_path: str | Path = "feature_importance.png",
):
    """
    Create and save the Random Forest feature-importance chart.
    """

    try:
        importance_df = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }
        ).sort_values(
            by="Importance",
            ascending=True,
        )

        output_path = Path(output_path)

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
        )

        ax.set_title("Real Estate Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        fig.tight_layout()
        fig.savefig(output_path)

        plt.close(fig)

        logger.info(
            "Feature-importance chart saved to %s",
            output_path,
        )

    except Exception as exc:
        logger.exception(
            "Feature-importance visualization failed."
        )

        raise RuntimeError(
            f"Unable to create feature-importance chart: {exc}"
        ) from exc