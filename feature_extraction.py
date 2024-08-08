import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    log_loss,
)
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from loguru import logger
from lib.data_prep import *
from sklearn.model_selection import train_test_split
from lib.crimeCategories import categorize_crime

RANDOM_SEED = 42

if __name__ == "__main__":

    data_sample = load_data()

    data_sample = data_sample.sample(n=300000, random_state=RANDOM_SEED)

    print(data_sample.head())

    print(data_sample.info())

    data_sample = format_data_frame(data_sample)

    data_sample = filter_outside_points(data_sample)

    logger.info(f"Grouping Categories")

    data_sample["Crime Categorie"] = data_sample["CrmCd.Desc"].apply(
        categorize_crime)

    features: pd.DataFrame = data_sample[
        [
            "AREA",
            "TIME.OCC",
            "Latitude",
            "Longitude",
            "SEASON",
            "WEEKDAY",
            "DATE.OCC.Month",
            "RD",
            "Street Category"
        ]
    ]

    target = data_sample["Crime Categorie"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[["Latitude", "Longitude"]])

    features.loc[:, ["Latitude", "Longitude"]] = scaled_features

    features = pd.get_dummies(features, columns=["AREA"])
    features = pd.get_dummies(features, columns=["SEASON"])
    features = pd.get_dummies(features, columns=["WEEKDAY"])
    features = pd.get_dummies(features, columns=["DATE.OCC.Month"])
    features = pd.get_dummies(features, columns=["Street Category"])
    features = pd.get_dummies(features, columns=["RD"])

    logger.info("---------------------------------------------")
    logger.info("Data-Preparation finished ...")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=RANDOM_SEED
    )

    model = DecisionTreeClassifier(
        max_depth=15,
        random_state=RANDOM_SEED,
    )
    min_features_to_select = 5
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy',
                  min_features_to_select=min_features_to_select)

    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.support_]

    logger.info(f"Selected Features: {selected_features}")

    model.fit(X_train[selected_features], y_train)

    feature_names = features.columns.tolist()

    class_names = target.unique().tolist()

    y_pred = model.predict(X_test[selected_features])

    y_pred_prob_tree = model.predict_proba(X_test[selected_features])

    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=selected_features,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    top_5_features = feature_importances.head(5).index.tolist()

    feature_importances_str = feature_importances.to_string()

    logger.info(f"Feature Importances:\n{feature_importances_str}")

    logger.info("Confusion Matrix:")
    accuracy = accuracy_score(y_test, y_pred)

    logger.success(f"Accuracy: {accuracy:.4f}")
    log_loss_tree = log_loss(y_test, y_pred_prob_tree)
    logger.success(f"Decision Tree Log Loss: {log_loss_tree}")
