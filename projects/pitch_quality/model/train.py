import json
import os
from datetime import datetime
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

import baseball.projects.pitch_quality.model.constants as mc
import baseball.utils.validation as val
from baseball.projects.pitch_quality.data.constants import CATEGORICAL_RESPONSE_INDICES, LOG_PROB_OFFSET_COLNAMES
from baseball.projects.pitch_quality.data.etl import load_pitch_quality_model_data, partition_pitch_data
from baseball.projects.pitch_quality.model.submodels.season_offsets import (
    CategoricalLogOffsets,
    load_categorical_log_offsets,
)

# from baseball.projects.pitch_quality.model.submodels.expected_movement import (
#     ExpectedMovement,
#     load_expected_movement_models,
# )
from baseball.utils.general import write_pickled_object
from baseball.utils.tuning import GPHPTuner

TODAY = datetime.now()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PATH = os.path.join(SCRIPT_DIR, "runs")

# did a tuning run, it liked these, so defaulting here if you don't want to re-tune
DEFAULT_PITCH_OUTCOME_PARAMS = {
    "eta": 0.3,
    "gamma": 10,
    "max_depth": 5,
    "min_child_weight": 10,
    "n_estimators": 173,
    "subsample": 0.8,
}


def _format_date_for_file() -> str:
    """Nicely(ish) formats a timestamp for use as a folder"""
    return str(TODAY).split(".")[0].replace(" ", "_").replace(":", ".")


def evaluate_pitch_outcome_predictions(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bm_train: Optional[np.ndarray] = None,
    bm_test: Optional[np.ndarray] = None,
    filepath: str = "",
) -> None:
    """ """
    # bare bones scoring
    scoring = val.score_classification(
        yhat_proba_train=model.predict_proba(X_train, base_margin=bm_train),
        y_train=y_train,
        yhat_proba_test=model.predict_proba(X_test, base_margin=bm_test),
        y_test=y_test,
    )
    print("-" * 50 + "\nModel Scoring:")
    print(scoring)
    scoring.to_csv(os.path.join(filepath, "scoring.csv"))

    def _savefig(filename: str) -> None:
        """ """
        plt.savefig(os.path.join(filepath, filename))
        plt.clf()

    for strategy in ("uniform", "quantile"):
        # training set calibration curve
        val.plot_calibration_curve_multiclass(
            y_true=y_train,
            y_prob=model.predict_proba(X_train, base_margin=bm_train),
            strategy=strategy,
            classes=list(CATEGORICAL_RESPONSE_INDICES.keys()),
        )
        _savefig(f"training-calibration-{strategy}.png")

        # test set calibration curve
        val.plot_calibration_curve_multiclass(
            y_true=y_test,
            y_prob=model.predict_proba(X_test, base_margin=bm_test),
            strategy=strategy,
            classes=list(CATEGORICAL_RESPONSE_INDICES.keys()),
        )
        _savefig(f"testing-calibration-{strategy}.png")


def train_pitch_outcome_model(
    run_type: Literal["train", "train_test"],
    tune_model: bool = True,
    load_from_cache: bool = True,
) -> None:
    """ """
    # pull in the data...presumption is you've cached it before
    pitch_data_df = load_pitch_quality_model_data(load_from_cache=load_from_cache).dropna(subset=mc.FEATURES)
    print("Pitch data loaded.")

    # add in categorical log offsets for season / environment
    offsets = load_categorical_log_offsets().predict(pitch_data_df)
    pitch_data_df[LOG_PROB_OFFSET_COLNAMES] = offsets
    print("Log offsets for season / environment loaded and added to `pitch_data_df")

    # TODO: fool around with this some more
    # xmvmt_models = load_expected_movement_models()
    # print("xMovement models loaded.")
    # pitch_data_df = xmvmt_models.predict(pitch_data_df)
    # print("xMovemented predicted.")

    pitch_data_df_train, pitch_data_df_test_player, pitch_data_df_test_season = partition_pitch_data(pitch_data_df)

    # training design + response
    X_train = pitch_data_df_train[mc.FEATURES].values.astype(float)
    y_train = pitch_data_df_train[mc.CATEGORICAL_RESPONSE].values.astype(int)
    bm_train = pitch_data_df_train[LOG_PROB_OFFSET_COLNAMES].values.astype(float)

    if tune_model:
        # instantiate tuning object
        tuner = GPHPTuner(
            estimator=XGBClassifier(random_state=10),
            # TODO: constants here
            params_continuous=dict(
                eta=uniform(0.1, 0.5),
                subsample=uniform(0.6, 0.4),
                gamma=uniform(0, 10),
                min_child_weight=uniform(1, 99),
            ),
            # TODO: constants here
            params_discrete=dict(max_depth=uniform(2.0, 6.5), n_estimators=uniform(10, 240)),
            loss_fn=log_loss,
            kfold=GroupKFold,
            cv=8,
            random_states=dict(kfold=100, random_search=101),
        )
        tuner.fit(
            # TODO: more args in here
            X_train,
            y_train,
            groups=pitch_data_df_train["pitcher"].values.astype(int),
            n_random_burn_in=100,
            n_gp_iter=20,
        )
        params = tuner.get_best_params()
    else:
        params = DEFAULT_PITCH_OUTCOME_PARAMS

    # refit the model on all the in-sample data
    model = XGBClassifier(**params, random_state=33)
    model.fit(X_train, y_train, base_margin=bm_train)

    # make the run path: you'll save relevant objects and results in here
    run_path = os.path.join(RUN_PATH, "pitch_outcome", f"{run_type}_" + _format_date_for_file())
    os.mkdir(run_path)

    # if you're testing, do a full on test run
    if run_type == "train_test":
        # test by OOS season and OOS Player
        for test_set in ("player", "season"):

            # which test are we using?
            test_df = pitch_data_df_test_player if test_set == "player" else pitch_data_df_test_season

            # design + response matrices?
            X_test = test_df[mc.FEATURES].values.astype(float)
            y_test = test_df[mc.CATEGORICAL_RESPONSE].values.astype(int)
            bm_test = test_df[LOG_PROB_OFFSET_COLNAMES].values.astype(float)

            # make a filepath for the run
            run_path_set = os.path.join(run_path, test_set)
            os.mkdir(run_path_set)

            # save sscoring + diagnostics
            evaluate_pitch_outcome_predictions(
                model, X_train, y_train, X_test, y_test, filepath=run_path_set, bm_train=bm_train, bm_test=bm_test
            )
    else:
        # otherwise, just score insample (twice...easier that way)
        evaluate_pitch_outcome_predictions(
            model, X_train, y_train, X_train, y_train, filepath=run_path, bm_train=bm_train, bm_test=bm_test
        )

    # save the model object
    write_pickled_object(model, os.path.join(run_path, "model.pkl"))

    # save the input features
    write_pickled_object(mc.FEATURES, os.path.join(run_path, "features.pkl"))

    # save the hyperparams
    with open(os.path.join(run_path, "params.json"), "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    # save tuning results, if desired
    if tune_model:
        # save the tuning object + summary
        write_pickled_object(tuner, os.path.join(run_path, "tuner.pkl"))
        tuner.fit_summary().to_csv(os.path.join(run_path, "tuning.csv"), index=False)


if __name__ == "__main__":
    train_pitch_outcome_model("train_test", tune_model=False)
