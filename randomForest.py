import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from loguru import logger
from lib.data_prep import *
import random
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from lib.tune_random_forest import (
    grid_tune_hyperparameters,
    randomize_tune_hyperparameters,
)
import gc
import optuna
import psutil
from lib.crimeCategories import crime_categories, categorize_crime
import seaborn as sns
import matplotlib.pyplot as plt


# RANDOM_SEED = random.randint(1, 10)  # können wir final setten zum Schluss
RANDOM_SEED = 41


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(
        f"RSS: {mem_info.rss / (1024 * 1024)          :.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB"
    )


def bayesian_optimization_forest(trial, X_train, y_train):

    n_estimators = trial.suggest_int("n_estimators", 40, 50)

    max_depth = trial.suggest_int("max_depth", 10, 50)

    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 300, step=10)

    min_samples_split = trial.suggest_int("min_samples_split", 5, 200)

    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 100)

    max_features = trial.suggest_categorical("max_features", ["sqrt", 0.7, None])

    bootstrap = True

    oob_score = True

    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_leaf_nodes=max_leaf_nodes,
        oob_score=oob_score,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        criterion=criterion,
        n_jobs=4,  # meiner schmiert ab mit -1
        verbose=1,
    )
    gc.collect()
    print_memory_usage()

    return cross_val_score(clf, X_train, y_train, cv=2, n_jobs=4).mean()


# RSS: 386.99 MB, VMS: 1333.82 MB ohne gc.collect und n = 200000
# kleiner Durchlauf:
"""
Mit Max Feature

Best parameters:
{'n_estimators': 90,
    'max_depth': 33,
    'max_leaf_nodes': 230,
    'min_samples_split': 10,
    'min_samples_leaf': 10,
    'max_features': None,
    'class_weight': None,
    'criterion': 'gini'}
 Best score: 0.5332957738701816

"""

"""
Best parameters: {
'n_estimators': 36,
 'max_depth': 39,
 'max_leaf_nodes': 180,
 'min_samples_split': 199,
 'min_samples_leaf': 35,
 'max_features': None,
 'class_weight': None,
 'criterion': 'gini'}
"""

"""
letztes Tuning
Best parameters: {'n_estimators': 30,
 'max_depth': 49,
   'max_leaf_nodes': 300,
     'min_samples_split': 50,
       'min_samples_leaf': 92,
         'max_features': None,
           'class_weight': None,
             'criterion': 'gini'}
Best score: 0.52086466478169

"""
if __name__ == "__main__":

    tuning: bool = False
    eval: bool = False

    start: str = input("Mit Tuning? -> Enter (Y) ")

    if start == "Y" or start == "y":
        tuning: bool = True

    start_eval: str = input("K-Fold-Cross-Validation andwenden ? -> (Y/y) ")

    if start_eval == "Y" or start_eval == "y":
        eval = True

    data_sample = load_data()

    print(data_sample.head())

    print(data_sample.info())

    # data_sample = data_sample.sample(n=700000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)
    data_sample = filter_outside_points(data_sample)
    logger.info(f"Grouping Categories")
    data_sample["Crime Categorie"] = data_sample["CrmCd.Desc"].apply(categorize_crime)

    # del data
    # gc.collect()

    features: pd.DataFrame = data_sample[
        [
            "AREA",
            "TIME.OCC",
            "Latitude",
            "Longitude",
            "SEASON",
            "WEEKDAY",
            "DATE.OCC.Year",
            "DATE.OCC.Month",
            #   'Diff between OCC and Report',
            #   'Status',
            #   'RD'
        ]
    ]

    target = data_sample["Crime Categorie"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[["Latitude", "Longitude"]])

    features.loc[:, ["Latitude", "Longitude"]] = scaled_features

    features = pd.get_dummies(features, columns=["AREA"])
    features = pd.get_dummies(features, columns=["SEASON"])
    features = pd.get_dummies(features, columns=["DATE.OCC.Year"])
    features = pd.get_dummies(features, columns=["DATE.OCC.Month"])
    features = pd.get_dummies(features, columns=["WEEKDAY"])
    # features = pd.get_dummies(features, columns=['Status'])
    # features = pd.get_dummies(features, columns=['RD'])

    logger.info("---------------------------------------------")
    logger.info("Data-Preparation finished ...")
    logger.info("Summary statistics after standardization:")
    logger.info(features.describe())
    logger.info(features.info())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.5, random_state=RANDOM_SEED
    )

    logger.info("---------------------------------------------")
    logger.info("Training the model ...")

    if tuning:
        logger.info("You chose to tune the model. This will take a moment...")
        # Use Optuna to tune the hyperparameters
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: bayesian_optimization_forest(trial, X_train, y_train),
            n_trials=300,
            n_jobs=-1,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")
        best_model = RandomForestClassifier(
            **best_params, random_state=RANDOM_SEED, n_jobs=4, verbose=2
        )

    ########################################################

    if not tuning:
        best_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=15,
            min_samples_split=45,
            max_features=None,
            max_depth=49,  # mein PC schmiert sonst ab
            max_leaf_nodes=300,
            random_state=RANDOM_SEED,
            verbose=2,
            bootstrap=True,
            oob_score=True,
            class_weight=None,
            criterion="gini",
            n_jobs=-1,
        )
    """
    letztes Tuning
    Best parameters: {'n_estimators': 30,
     'max_depth': 44,
       'max_leaf_nodes': 300,
         'min_samples_split': 50,
           'min_samples_leaf': 92,
             'max_features': None,
               'class_weight': None,
                 'criterion': 'gini'}
    Best score: 0.52086466478169
    """

    """
    nacht tuning
    Best parameters: {'n_estimators': 47,
      'max_depth': 49,
        'max_leaf_nodes': 300,
          'min_samples_split': 45,
            'min_samples_leaf': 15,
              'max_features': None,
                'class_weight': None,
                  'criterion': 'gini'}
    0.5213990999262593

    """

    if eval:
        logger.info("Doing a precise evaluation of the model")
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="weighted"),
            "recall": make_scorer(recall_score, average="weighted"),
            "f1": make_scorer(f1_score, average="weighted"),
            "log_loss": make_scorer(log_loss, needs_proba=True),
        }

        # Durchführung der Cross-Validation
        # Verwendung von StratifiedKFold für Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        # Durchführung der Cross-Validation
        cv_results = cross_validate(
            best_model,
            features,
            target,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
        )

        for metric in scoring.keys():
            mean_score = np.mean(cv_results[f"test_{metric}"])
            std_score = np.std(cv_results[f"test_{metric}"])
            logger.info(
                f"{metric.capitalize()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}"
            )

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        folds = np.arange(1, 6)
        accuracies = cv_results["test_accuracy"]

        plt.plot(
            folds, accuracies, marker="o", linestyle="-", color="b", label="Accuracy"
        )
        plt.fill_between(
            folds,
            accuracies - np.std(accuracies),
            accuracies + np.std(accuracies),
            color="b",
            alpha=0.2,
        )

        plt.title("Cross-Validation Accuracy per Fold", fontsize=16)
        plt.xlabel("Fold", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xticks(folds)
        plt.legend()
        os.makedirs("Plots", exist_ok=True)
        plt.savefig("Plots/CV_Forest.png")
        logger.info("CV Forest  Results saved")

        # Das Modell auf den gesamten Datensatz anpassen
        # best_model.fit(features, target)
        # oob_score = best_model.oob_score_  # Berechnung des OOB-Scores
        # logger.info(f'OOB Score: {oob_score:.4f}')  # Ausgabe des OOB-Scores

    if not eval:
        # Modell trainieren
        best_model.fit(X_train, y_train)

        del X_train, y_train
        gc.collect()

        # Modellbewertung
        test_score = best_model.score(X_test, y_test)
        logger.info(f"Test-Set Score: {test_score}")

        logger.info("---------------------------------------------")

        logger.info("Predicting the model ...")
        y_pred = best_model.predict(X_test)
        y_pred_prob_rf = best_model.predict_proba(X_test)

        logger.info("---------------------------------------------")
        classification_rep = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{classification_rep}")

        feature_importances = pd.DataFrame(
            best_model.feature_importances_,
            index=features.columns,
            columns=["importance"],
        ).sort_values("importance", ascending=False)
        feature_importances_str = feature_importances.to_string()
        # Nutzung von `logger` um die vollständige Liste zu loggen
        logger.info(f"Feature Importances:\n{feature_importances_str}")

        logger.success(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        unique_classes = np.unique(y_test)

        log_loss_rf = log_loss(y_test, y_pred_prob_rf, labels=unique_classes)

        logger.success(f"Random Forest Log Loss: {log_loss_rf}")
