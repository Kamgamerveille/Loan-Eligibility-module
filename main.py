import logging

from src.data.make_dataset import (
    load_and_clean_data
)

from src.features.build_features import (
    build_features
)

from src.models.train_model import (
    train_models
)

from src.visualization.visualize import (
    plot_loan_distribution,
    plot_model_comparison,
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

    try:
        print(
            "\nStarting Loan Eligibility Project..."
        )

        # Load and clean data
        df = load_and_clean_data(
            "data/raw/credit.csv"
        )

        # Create first visualization
        plot_loan_distribution(
            df
        )

        # Build model features
        X, y = build_features(
            df
        )

        # Train models
        training_results = train_models(
            X,
            y
        )

        # Model comparison chart
        plot_model_comparison(
            training_results[
                "results"
            ]
        )

        print(
            "\nMODEL RESULTS"
        )

        print(
            "---------------------------"
        )

        for name, accuracy in (
            training_results[
                "results"
            ].items()
        ):

            print(
                f"{name}: "
                f"{accuracy:.2%}"
            )

        print(
            "\nBest Model:",
            training_results[
                "model_name"
            ]
        )

        print(
            "Best Accuracy:",
            f"{training_results['accuracy']:.2%}"
        )

        print(
            "\nModel saved successfully."
        )

    except Exception as exc:

        logger.exception(
            "Loan Eligibility pipeline failed."
        )

        print(
            f"Pipeline error: {exc}"
        )


if __name__ == "__main__":
    main()