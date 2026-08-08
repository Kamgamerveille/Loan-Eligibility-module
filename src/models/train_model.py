import logging
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


logger = logging.getLogger(__name__)


def train_models(
    X,
    y,
    model_path="models/LoanEligibilityModel.pkl",
    scaler_path="models/scaler.pkl",
    columns_path="models/model_columns.pkl",
):
    """
    Train Logistic Regression, Decision Tree,
    and Random Forest models.

    Select and save the best model.
    """

    try:
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        models = {
            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
        }

        results = {}

        best_model = None
        best_name = None
        best_accuracy = 0

        for name, model in models.items():

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            results[name] = accuracy

            logger.info(
                "%s accuracy: %.4f",
                name,
                accuracy
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

        # Create model directory
        model_path = Path(model_path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save best model
        with open(
            model_path,
            "wb"
        ) as model_file:
            pickle.dump(
                best_model,
                model_file
            )

        # Save scaler
        with open(
            scaler_path,
            "wb"
        ) as scaler_file:
            pickle.dump(
                scaler,
                scaler_file
            )

        # Save exact columns used during training
        with open(
            columns_path,
            "wb"
        ) as columns_file:
            pickle.dump(
                list(X.columns),
                columns_file
            )

        logger.info(
            "Best model: %s with accuracy %.4f",
            best_name,
            best_accuracy
        )

        return {
            "model": best_model,
            "model_name": best_name,
            "accuracy": best_accuracy,
            "results": results,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_test_scaled": X_test_scaled,
        }

    except Exception as exc:
        logger.exception(
            "Model training failed."
        )

        raise RuntimeError(
            f"Unable to train models: {exc}"
        ) from exc