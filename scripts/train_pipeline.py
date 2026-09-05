"""Tune, refit, and serialize the deployment house-price pipeline."""

from __future__ import annotations

from src.config import PROCESSED_TRAIN_PATH
from src.data.preprocessing import load_processed_data
from src.models.predict import save_pipeline
from src.models.train import (
    create_train_test_split,
    refit_pipeline_on_all_data,
    split_features_and_target,
    tune_gradient_boosting_model,
)


def main() -> None:
    data = load_processed_data(PROCESSED_TRAIN_PATH)
    X_all, y_all = split_features_and_target(data)
    X_train, _, y_train, _ = create_train_test_split(data)

    search = tune_gradient_boosting_model(X_train, y_train)
    deployment_pipeline = refit_pipeline_on_all_data(search.best_estimator_, X_all, y_all)
    artifact_path = save_pipeline(deployment_pipeline)

    print(f"Best cross-validation RMSE: ${-search.best_score_:,.2f}")
    print(f"Saved deployment pipeline to {artifact_path}.")


if __name__ == "__main__":
    main()
