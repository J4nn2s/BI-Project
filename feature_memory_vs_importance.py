import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42


def plot_feature_importance_with_memory(dummy_groups: dict, importances: dict, memory_usage: dict) -> None:
    categories = list(dummy_groups.keys())
    memory_usage_values = [memory_usage[category] for category in categories]
    importance_values = [importances[category] for category in categories]

    # Skalieren der Feature-Importance-Werte
    max_memory_usage = max(memory_usage_values)
    scaled_importance_values = [
        value * max_memory_usage for value in importance_values]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    bar_width = 0.35
    index = range(len(categories))

    bar1 = ax1.bar(index, memory_usage_values, bar_width,
                   label='Speichernutzung (Bytes)', color='b', alpha=0.6)
    bar2 = ax1.bar([i + bar_width for i in index], scaled_importance_values,
                   bar_width, label='Feature Importance (skaliert)', color='g', alpha=0.6)

    ax1.set_xlabel('Kategorie')
    ax1.set_ylabel('Speichernutzung (Bytes) / Feature Importance (skaliert)')
    ax1.set_title('Speichernutzung und Feature Importance je Kategorie')
    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()

    plt.tight_layout()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/feature_importance_with_memory_side_by_side_scaled.png')
    logger.info(
        "Feature importance with memory usage saved as 'Plots/feature_importance_with_memory_side_by_side_scaled.png'")


def plot_feature_importance(dummy_groups: dict, importances: dict, memory_usage: dict) -> None:
    categories = list(dummy_groups.keys())
    num_columns = [len(columns) for columns in dummy_groups.values()]
    memory_usage_values = [memory_usage[category] for category in categories]
    importance_values = [importances[category] for category in categories]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    sns.barplot(x=categories, y=memory_usage_values,
                ax=axes[0], palette='deep')
    axes[0].set_title('Speichernutzung je Kategorie')
    axes[0].set_xlabel('Kategorie')
    axes[0].set_ylabel('Speichernutzung (Bytes)')
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=categories, y=importance_values, ax=axes[1], palette='deep')
    axes[1].set_title('Feature Importance je Kategorie')
    axes[1].set_xlabel('Kategorie')
    axes[1].set_ylabel('Feature Importance')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/feature_importance.png')
    logger.info("Feature importance saved as 'Plots/feature_importance.png'")


def cross_validated_importances(X, y, dummy_groups, cv=10, random_state=42):
    """
    Führt eine 10-fache Cross-Validation durch und berechnet die mittlere Feature-Importance für jede Gruppe.

    Parameters:
    - X: Feature-Matrix (numpy array oder pandas DataFrame)
    - y: Zielvariable (numpy array oder pandas Series)
    - dummy_groups: Dictionary mit den Gruppennamen als Schlüssel und Listen der zugehörigen Dummy-Variablen als Werte.
    - cv: Anzahl der Folds für die Cross-Validation (Standard: 10).
    - random_state: Zufallsseed für die Reproduzierbarkeit.

    Returns:
    - group_importances_mean: Durchschnittliche Feature-Importance für jede Gruppe über alle Folds hinweg.
    """

    model = DecisionTreeClassifier(max_depth=15, random_state=random_state)
    # Matrix zur Speicherung der Importances für jeden Fold
    importances = np.zeros((cv, len(X.columns)))

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        importances[i, :] = model.feature_importances_

    # Durchschnittliche Feature-Importance für jede Gruppe berechnen
    group_importances_mean = {}
    for group, columns in dummy_groups.items():
        group_importances_mean[group] = np.mean(
            importances[:, [X.columns.get_loc(col) for col in columns]], axis=1).mean()

    return group_importances_mean


if __name__ == "__main__":

    data_sample = load_data_train()

    data_sample = data_sample.sample(n=100000, random_state=RANDOM_SEED)

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
            "day_of_month",
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
    features = pd.get_dummies(features, columns=["day_of_month"])

    logger.info("---------------------------------------------")
    logger.info("Data-Preparation finished ...")

    dummy_groups = {
        'AREA': [col for col in features.columns if col.startswith('AREA')],
        'SEASON': [col for col in features.columns if col.startswith('SEASON')],
        'WEEKDAY': [col for col in features.columns if col.startswith('WEEKDAY')],
        'DATE.OCC.Month': [col for col in features.columns if col.startswith('DATE.OCC.Month')],
        'Street Category': [col for col in features.columns if col.startswith('Street Category')],
        'RD': [col for col in features.columns if col.startswith('RD')],
        'day_of_month': [col for col in features.columns if col.startswith('day_of_month')],
        'TIME.OCC': ['TIME.OCC'],
        'Latitude': ['Latitude'],
        'Longitude': ['Longitude']

    }
    importances = {
        'AREA': 0.0013287403922143683,
        'SEASON': 0.007231677556350594,
        'WEEKDAY': 0.008344290855425235,
        'DATE.OCC.Month': 0.005586113853804397,
        'Street Category': 0.0018813851991367554,
        'RD': 0.00010150453734319263,
        'day_of_month': 0.003635070063926071,
        'TIME.OCC': 0.26113598210215516,
        'Latitude': 0.13569565283640167,
        'Longitude': 0.1407342175897967
    }
    memory_usage = {
        'AREA': 2898985,
        'SEASON': 1199580,
        'WEEKDAY': 1499475,
        'DATE.OCC.Month': 1999300,
        'Street Category': 3398810,
        'RD': 117558840,
        'day_of_month': 3898635,
        'TIME.OCC': 1599440,
        'Latitude': 1199580,
        'Longitude': 1199580
    }

    plot_feature_importance(dummy_groups, importances, memory_usage)
    plot_feature_importance_with_memory(
        dummy_groups, importances, memory_usage)
    # for category, columns in dummy_groups.items():
    #     num_columns = len(columns)
    #     memory_usage = features[columns].memory_usage(deep=True).sum()
    #     logger.info(f"{category} nimmt {num_columns} Spalten ein und verbraucht {
    #                 memory_usage} Bytes Speicher.")
    # logger.info("Starting Backward Selection with Groups")

    # group_importances = cross_validated_importances(
    #     features, target, dummy_groups)

    # for group, importance in group_importances.items():
    #     logger.info(f"{group}: {importance}\n")
