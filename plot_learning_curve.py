from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    StratifiedKFold,
    KFold,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
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
from sklearn.model_selection import train_test_split
import gc
import optuna
import matplotlib.pyplot as plt
import re
from lib.crimeCategories import crime_categories, categorize_crime
import seaborn as sns


RANDOM_SEED = 42

models = {
    # "Decision Tree": DecisionTreeClassifier(
    #     max_depth=15,
    #     min_samples_split=20,
    #     min_samples_leaf=9,
    #     max_features=None,
    #     criterion="gini",
    #     random_state=RANDOM_SEED
    # ),
    "Random Forest": RandomForestClassifier(n_estimators=50,
                                            min_samples_leaf=35,
                                            min_samples_split=199,
                                            max_features=None,
                                            max_depth=39,
                                            max_leaf_nodes=230,
                                            random_state=RANDOM_SEED,
                                            verbose=2,
                                            bootstrap=True,
                                            oob_score=True,
                                            class_weight=None,
                                            criterion="gini",
                                            n_jobs=4)
}

# Best parameters: {
# 'n_estimators': 36,
#  'max_depth': 39,
#  'max_leaf_nodes': 180,
#  'min_samples_split': 199,
#  'min_samples_leaf': 35,
#  'max_features': None,
#  'class_weight': None,
#  'criterion': 'gini'}


def plot_learning_curve_no_cv(
    estimator, title, features, target, train_sizes=np.linspace(0.01, 1.0, 10)
):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        features,
        target,
        cv=None,
        train_sizes=train_sizes,
        random_state=RANDOM_SEED,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.ylim(0, 1)  # Set the y-axis range from 0 to 1

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean,
             "o-", color="g", label="Test score")

    plt.legend(loc="best")
    os.makedirs("Plots", exist_ok=True)
    model_name_clean = re.sub(r"\W+", "", title)
    plt.savefig(f"Plots/Learning_Curve_{model_name_clean}.png")
    logger.info(
        f"Learning Curve for {
            title} saved as Plots/Learning_Curve_{model_name_clean}.png"
    )


def plot_learning_curve(
    estimator,
    title,
    features,
    target,
    cv=10,
    n_jobs=-1,
    train_sizes=np.linspace(0.01, 1.0, 10),
):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        features,
        target,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        random_state=RANDOM_SEED,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.ylim(0, 1)  # Set the y-axis range from 0 to 1

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    os.makedirs("Plots", exist_ok=True)
    model_name_clean = re.sub(r"\W+", "", title)
    plt.savefig(f"Plots/Learning_Curve_{model_name_clean}.png")
    logger.info(
        f"Learning Curve for {
            title} saved as Plots/Learning_Curve_{model_name_clean}.png"
    )


if __name__ == "__main__":
    data_sample = load_data_train()

    print(data_sample.head())
    print(data_sample.info())

    data_sample = data_sample.sample(n=500000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)

    data_sample = filter_outside_points(data_sample)

    logger.info(f"Grouping Categories")
    data_sample["Crime Categorie"] = data_sample["CrmCd.Desc"].apply(
        categorize_crime)

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
            "DATE.OCC.Month",
            "day_of_month"
            #   'Diff between OCC and Report',
            #   'Status',
            #   'RD',
            # "DATE.OCC.Year",
        ]
    ]

    target = data_sample["Crime Categorie"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[["Latitude", "Longitude"]])

    features.loc[:, ["Latitude", "Longitude"]] = scaled_features

    features = pd.get_dummies(features, columns=["AREA"])
    features = pd.get_dummies(features, columns=["SEASON"])
    features = pd.get_dummies(features, columns=["DATE.OCC.Month"])
    features = pd.get_dummies(features, columns=["WEEKDAY"])
    features = pd.get_dummies(features, columns=["day_of_month"])
    # features = pd.get_dummies(features, columns=['Status'])
    # features = pd.get_dummies(features, columns=['RD'])
    # features = pd.get_dummies(features, columns=["DATE.OCC.Year"])
    logger.info("Starting to plot the curve")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=RANDOM_SEED
    )

    logger.info("Starting to plot the learning curves")
    train_sizes = np.linspace(0.01, 1.0, 10)  # Von 1% bis 100% des Datensatzes

    logger.info(f"Hier die Train sizes{train_sizes}")

    for model_name, model in models.items():
        logger.info(f"Plotting {model_name}")
        logger.info(f"{model_name}")

        plot_learning_curve_no_cv(
            model,
            f"{model_name}",
            X_train,
            y_train,
            train_sizes=train_sizes
        )

        # plot_learning_curve(model, f"Learning Curve for {model_name}", features, target, cv=10, train_sizes=train_sizes
        # )
