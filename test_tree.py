import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
from lib.crimeCategories import categorize_crime

if __name__ == "__main__":
    RANDOM_SEED: int = 42

    data_train: pd.DataFrame = load_data_train()
    data_test: pd.DataFrame = load_data_test()

    data_train = format_data_frame(data_train)
    data_test = format_data_frame(data_test)

    data_train = filter_outside_points(data_train)
    data_test = filter_outside_points(data_test)

    logger.info(f"Grouping Categories")

    data_train["Crime Categorie"] = data_train["CrmCd.Desc"].apply(
        categorize_crime)
    data_test["Crime Categorie"] = data_test["CrmCd.Desc"].apply(
        categorize_crime)

    model: DecisionTreeClassifier = DecisionTreeClassifier(
        max_depth=14,
        min_samples_split=38,
        min_samples_leaf=72,
        max_features=None,
        criterion="gini",
        random_state=RANDOM_SEED,
    )

    features_train: pd.DataFrame = data_train[
        [
            "AREA",
            "TIME.OCC",
            "Latitude",
            "Longitude",
            "SEASON",
            "WEEKDAY",
            "DATE.OCC.Month",
            # "Status" ,
            # "DATE.OCC.Year" ,
            # "Diff between OCC and Report" ,
        ]
    ]
    features_test: pd.DataFrame = data_test[
        [
            "AREA",
            "TIME.OCC",
            "Latitude",
            "Longitude",
            "SEASON",
            "WEEKDAY",
            "DATE.OCC.Month",
            # "Status" ,
            # "DATE.OCC.Year" ,
            # "Diff between OCC and Report" ,
        ]
    ]

    target_train = data_train["Crime Categorie"]
    target_test = data_test["Crime Categorie"]

    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(
        features_train[["Latitude", "Longitude"]])
    scaled_features_test = scaler.fit_transform(
        features_test[["Latitude", "Longitude"]])

    features_train.loc[:, ["Latitude", "Longitude"]] = scaled_features_train
    features_test.loc[:, ["Latitude", "Longitude"]] = scaled_features_test

    features_train = pd.get_dummies(features_train, columns=["AREA"])
    features_train = pd.get_dummies(features_train, columns=["SEASON"])
    features_train = pd.get_dummies(features_train, columns=["WEEKDAY"])
    features_train = pd.get_dummies(features_train, columns=["DATE.OCC.Month"])

    features_test = pd.get_dummies(features_test, columns=["AREA"])
    features_test = pd.get_dummies(features_test, columns=["SEASON"])
    features_test = pd.get_dummies(features_test, columns=["WEEKDAY"])
    features_test = pd.get_dummies(features_test, columns=["DATE.OCC.Month"])

    logger.info("---------------------------------------------")

    logger.info("Data-Preparation finished ...")

    logger.info("Start to build model")

    model.fit(features_train, target_train)

    y_pred = model.predict(features_test)
    y_pred_prob_tree = model.predict_proba(features_test)

    logger.info("Classification Report:")
    logger.info(classification_report(target_test, y_pred))

    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=features_train.columns,
        columns=["importance"],
    ).sort_values("importance", ascending=False)
    top_5_features = feature_importances.head(5).index.tolist()

    feature_importances_str = feature_importances.to_string()
    # Nutzung von `logger` um die vollständige Liste zu loggen
    logger.info(f"Feature Importances:\n{feature_importances_str}")

    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(target_test, y_pred))
    accuracy = accuracy_score(target_test, y_pred)
    logger.success(f"Accuracy: {accuracy:.4f}")
    log_loss_tree = log_loss(target_test, y_pred_prob_tree)
    logger.success(f"Decision Tree Log Loss: {log_loss_tree}")
