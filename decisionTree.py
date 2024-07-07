import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from loguru import logger
from lib.data_prep import *
import random
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from lib.tune_random_forest import grid_tune_hyperparameters, randomize_tune_hyperparameters
import gc
import optuna

# RANDOM_SEED = random.randint(1, 10)  # können wir final setten zum Schluss
RANDOM_SEED = 41


def bayesian_optimization(trial, X_train, y_train):
    max_depth = trial.suggest_int('max_depth', 10, 40)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical(
        'max_features', [None, 'sqrt', 'log2'])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=RANDOM_SEED
    )

    return cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1).mean()


if __name__ == "__main__":
    tuning: bool = False

    start: str = input("Mit Tuning? -> Enter (Y) ")

    if start == "Y" or start == "y":
        tuning: bool = True

    data_sample = load_data()

    print(data_sample.head())
    print(data_sample.info())

    # data_sample = data.sample(n=900000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)
    data_sample = remove_outside_la(data_sample)

    # del data
    # gc.collect()

    features: pd.DataFrame = data_sample[['AREA',
                                          'TIME.OCC',
                                          'Latitude',
                                          'Longitude',
                                          'SEASON',
                                          'WEEKDAY',
                                          'DATE.OCC.Year',
                                          'Diff between OCC and Report',
                                          'RD',
                                          'Status']]

    target = data_sample['Crm.Cd']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features[['Latitude', 'Longitude']])

    features.loc[:, ['Latitude', 'Longitude']] = scaled_features

    features = pd.get_dummies(features, columns=['AREA'])
    features = pd.get_dummies(features, columns=['SEASON'])
    features = pd.get_dummies(features, columns=['DATE.OCC.Year'])
    features = pd.get_dummies(features, columns=['WEEKDAY'])
    features = pd.get_dummies(features, columns=['Status'])

    logger.info('---------------------------------------------')
    logger.info("Data-Preparation finished ...")
    logger.info("Summary statistics after standardization:")
    logger.info(features.describe())
    logger.info(features.head())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_SEED)

    if tuning:
        logger.info("Starting Bayesian Optimization")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: bayesian_optimization(
            trial, X_train, y_train), n_trials=50, n_jobs=1, show_progress_bar=True)

        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")

        # Train the best model
        best_model = DecisionTreeClassifier(
            **best_params, random_state=RANDOM_SEED)
        best_model.fit(X_train, y_train)

    del data_sample, target
    gc.collect()

    logger.info('---------------------------------------------')
    logger.info("Training the model ...")

    if not tuning:
        best_model = DecisionTreeClassifier(
            max_depth=13,
            min_samples_split=13,
            min_samples_leaf=2,
            max_features=None,
            criterion="gini",
            random_state=RANDOM_SEED)

    #######################################################

    '''
    :100 - Best parameters found: {'max_depth': 13,
    'min_samples_split': 13,
    'min_samples_leaf': 2,
        'max_features': None,
        'criterion': 'gini'}
    '''
    #######################################################

    # Modell trainieren
    best_model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    y_pred = best_model.predict(X_test)
    y_pred_prob_tree = best_model.predict_proba(X_test)

    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Konfusionsmatrix
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)

    # Ausgabe der Accuracy
    logger.success(f"Accuracy: {accuracy:.4f}")
    log_loss_tree = log_loss(y_test, y_pred_prob_tree)
    logger.success(f"Decision Tree Log Loss: {log_loss_tree}")
