import logging

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


def plot_loan_distribution(
    df,
    output_path="loan_distribution.png",
):
    """
    Plot approved vs denied loan applications.
    """

    try:
        counts = df[
            "Loan_Approved"
        ].value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Loan Approval Distribution"
        )

        ax.set_xlabel(
            "Loan Status"
        )

        ax.set_ylabel(
            "Number of Applications"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        fig.tight_layout()

        fig.savefig(
            output_path
        )

        plt.close(fig)

        logger.info(
            "Loan distribution chart saved."
        )

    except Exception as exc:
        logger.exception(
            "Loan distribution visualization failed."
        )

        raise RuntimeError(
            f"Unable to create chart: {exc}"
        ) from exc


def plot_model_comparison(
    results,
    output_path="model_comparison.png",
):
    """
    Compare model accuracy.
    """

    try:
        result_df = pd.DataFrame(
            {
                "Model": list(
                    results.keys()
                ),
                "Accuracy": list(
                    results.values()
                ),
            }
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            result_df["Model"],
            result_df["Accuracy"]
        )

        ax.set_title(
            "Model Accuracy Comparison"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_ylim(
            0,
            1
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        fig.tight_layout()

        fig.savefig(
            output_path
        )

        plt.close(fig)

        logger.info(
            "Model comparison chart saved."
        )

    except Exception as exc:
        logger.exception(
            "Model comparison visualization failed."
        )

        raise RuntimeError(
            f"Unable to create model comparison: {exc}"
        ) from exc