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
from sklearn.inspection import PartialDependenceDisplay


def replace_text(obj):
    if isinstance(obj, plt.Text):
        txt = obj.get_text()
        txt = re.sub(r"\svalue\s*=\s*\[[^\]]+\]", "", txt)
        obj.set_text(txt)
    return obj


def save_decision_tree_plot(
    model,
    feature_names,
    class_names,
    output_dir="Plots",
    filename="decision_tree_plot.png",
    max_depth=4,
    filled=True,
    fontsize=10,
    impurity=True,
    node_ids=False,
    label="all",
    proportion=False,
    tree_width=20,  # Vergrößerung der Kacheln
    tree_height=15,  # Vergrößerung der Kacheln
) -> None:  # Pfeil-Größe anpassen
    class_names = [str(name) for name in class_names]
    feature_names = [str(name) for name in feature_names]

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(tree_width, tree_height))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=filled,
        max_depth=max_depth,
        fontsize=fontsize,
        impurity=impurity,  # Gini-Wert anzeigen
        proportion=proportion,  # Proportion nicht anzeigen
        node_ids=node_ids,  # Node-IDs anzeigen
        label=label,
    )  # Pfeil-Stil anpassen

    ax = plt.gca()
    for text in ax.texts:
        replace_text(text)

    output_path = os.path.join(output_dir, filename)

    # Plot speichern
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Decision tree plot saved in {output_path}")


RANDOM_SEED = random.randint(1, 10)  # können wir final setten zum Schluss
# RANDOM_SEED = 42


def bayesian_optimization(trial, X_train, y_train):
    max_depth = trial.suggest_int("max_depth", 10, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 5, 200)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 100)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=RANDOM_SEED,
    )

    return cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1).mean()


if __name__ == "__main__":
    tuning: bool = False

    start: str = input("Mit Tuning? -> Enter (Y/y) ")

    if start == "Y" or start == "y":
        tuning: bool = True

    eval: bool = False

    start_eval: str = input("Mit Cross-Val Evaluation? (Y/y) ")
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
            "DATE.OCC.Month",
            "DATE.OCC.Year",
            "Diff between OCC and Report",
            #   'RD',
            #   'Street Category',
            "Status",
        ]
    ]

    target = data_sample["Crime Categorie"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[["Latitude", "Longitude"]])

    features.loc[:, ["Latitude", "Longitude"]] = scaled_features

    features = pd.get_dummies(features, columns=["AREA"])
    features = pd.get_dummies(features, columns=["SEASON"])
    features = pd.get_dummies(features, columns=["DATE.OCC.Year"])
    features = pd.get_dummies(features, columns=["WEEKDAY"])
    features = pd.get_dummies(features, columns=["DATE.OCC.Month"])
    features = pd.get_dummies(features, columns=["Status"])
    # features = pd.get_dummies(features, columns=['Street Category'])
    # features = pd.get_dummies(features, columns=['RD'])

    print(features.info())

    logger.info("---------------------------------------------")
    logger.info("Data-Preparation finished ...")
    logger.info("Summary statistics after standardization:")
    logger.info(features.describe())
    logger.info(features.head())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=RANDOM_SEED
    )

    if tuning:
        logger.info("Starting Bayesian Optimization")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: bayesian_optimization(trial, X_train, y_train),
            n_trials=40,
            n_jobs=-1,
            show_progress_bar=True,
        )

        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")

        # Train the best model
        best_model = DecisionTreeClassifier(**best_params, random_state=RANDOM_SEED)

    del data_sample
    gc.collect()

    logger.info("---------------------------------------------")
    logger.info("Training the model ...")

    if not tuning:
        best_model = DecisionTreeClassifier(
            max_depth=14,
            min_samples_split=38,
            min_samples_leaf=72,
            max_features=None,
            criterion="gini",
            random_state=RANDOM_SEED,
        )

    ############################################################
    # altes Tuning
    # :100 - Best parameters found: {'max_depth': 13,
    # 'min_samples_split': 13,
    # 'min_samples_leaf': 2,
    #     'max_features': None,
    #     'criterion': 'gini'}

    """
    neues Tuning
    Best parameters found: {
    'max_depth': 14,
      'min_samples_split': 20,
        'min_samples_leaf': 9,
          'max_features': None,
      'criterion': 'gini'}
    """
    """
  Best parameters found:
    {'max_depth': 25,
    'min_samples_split': 38,
    'min_samples_leaf': 72,
    'max_features': None,
    'criterion': 'entropy'}
    }


    """
    ############################################################

    # Modell trainieren
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
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

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
        folds = np.arange(1, 11)
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
        plt.savefig("Plots/CV_Tree.png")
        logger.info("CV Tree Results saved")

    if not eval:
        best_model.fit(X_train, y_train)
        feature_names = features.columns.tolist()
        class_names = target.unique().tolist()

        # max_depth kann optional angepasst werden
        save_decision_tree_plot(best_model, feature_names, class_names, max_depth=5)

        # del X_train, y_train, target
        # gc.collect()

        y_pred = best_model.predict(X_test)
        y_pred_prob_tree = best_model.predict_proba(X_test)

        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred))

        feature_importances = pd.DataFrame(
            best_model.feature_importances_,
            index=features.columns,
            columns=["importance"],
        ).sort_values("importance", ascending=False)
        top_5_features = feature_importances.head(5).index.tolist()

        # Erstellen des Partial Dependence Plots für die Top-5-Features
        classes = best_model.classes_

        fig, axes = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
        for i, target_class in enumerate(classes):
            row, col = divmod(i, 4)
            PartialDependenceDisplay.from_estimator(
                best_model,
                X_test,
                features=top_5_features,
                target=target_class,
                grid_resolution=50,
                ax=axes[row, col],
            )
            axes[row, col].set_title(f"Class {target_class}")

        # Anpassen des Layouts
        plt.suptitle(
            "Partial Dependence Plots for Top 5 Features Across All Classes",
            fontsize=16,
        )
        plt.subplots_adjust(top=0.95)  # Platz für den Titel schaffen

        os.makedirs("Plots", exist_ok=True)
        plt.savefig("Plots/Partial_Tree.png")
        logger.info("CV Tree Results saved")

        feature_importances_str = feature_importances.to_string()
        # Nutzung von `logger` um die vollständige Liste zu loggen
        logger.info(f"Feature Importances:\n{feature_importances_str}")

        # Konfusionsmatrix
        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)

        # Ausgabe der Accuracy
        logger.success(f"Accuracy: {accuracy:.4f}")
        log_loss_tree = log_loss(y_test, y_pred_prob_tree)
        logger.success(f"Decision Tree Log Loss: {log_loss_tree}")
