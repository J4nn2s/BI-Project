import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from loguru import logger
from lib.data_prep import *
import random
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from lib.tune_random_forest import grid_tune_hyperparameters, randomize_tune_hyperparameters
import gc
import optuna
import psutil
from lib.crimeCategories import crime_categories, categorize_crime


RANDOM_SEED = random.randint(1, 10)  # können wir final setten zum Schluss
# RANDOM_SEED = 41


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 * 1024)
          :.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB")


def bayesian_optimization_forest(trial, X_train, y_train):

    n_estimators = trial.suggest_int('n_estimators', 20, 50)

    max_depth = trial.suggest_int('max_depth', 10, 60)

    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 50, 300, step=10)

    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    max_features = trial.suggest_categorical(
        'max_features', [None, 'sqrt', 'log2', 0.5, 0.75])

    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    oob_score = trial.suggest_categorical('oob_score', [True, False])

    class_weight = trial.suggest_categorical(
        'class_weight', [None, 'balanced'])

    criterion = trial.suggest_categorical(
        'criterion', ['gini', 'entropy', 'log_loss'])

    if oob_score and not bootstrap:
        bootstrap = True

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        criterion=criterion,
        n_jobs=2,  # meiner schmiert ab mit -1
        verbose=2
    )
    gc.collect()
    print_memory_usage()

    return cross_val_score(clf, X_train, y_train, cv=5, n_jobs=2).mean()


# RSS: 386.99 MB, VMS: 1333.82 MB ohne gc.collect und n = 200000
# kleiner Durchlauf:
'''
Best parameters: {
    'n_estimators': 34,
    'max_depth': 60,
    'min_samples_split': 10,
    'min_samples_leaf': 8,
    'max_features': 0.5,
    'bootstrap': False,
    'oob_score': False,
    'class_weight': None}
'''

if __name__ == "__main__":

    tuning: bool = False

    start: str = input("Mit Tuning? -> Enter (Y) ")

    if start == "Y" or start == "y":
        tuning: bool = True

    data = load_data()

    print(data.head())

    print(data.info())

    data_sample = data.sample(n=200000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)
    data_sample = remove_outside_la(data_sample)
    logger.info(f"Grouping Categories")
    data_sample['Crime Categorie'] = data_sample['CrmCd.Desc'].apply(
        categorize_crime)

    del data
    gc.collect()

    features: pd.DataFrame = data_sample[['AREA',
                                          'TIME.OCC',
                                          'Latitude',
                                          'Longitude',
                                          'SEASON',
                                          'WEEKDAY',
                                          'DATE.OCC.Year',
                                          'DATE.OCC.Month',
                                          'Diff between OCC and Report',
                                          'Status',
                                          'RD']]

    target = data_sample['Crime Categorie']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features[['Latitude', 'Longitude']])

    features.loc[:, ['Latitude', 'Longitude']] = scaled_features

    features = pd.get_dummies(features, columns=['AREA'])
    features = pd.get_dummies(features, columns=['SEASON'])
    features = pd.get_dummies(features, columns=['DATE.OCC.Year'])
    features = pd.get_dummies(features, columns=['DATE.OCC.Month'])
    features = pd.get_dummies(features, columns=['WEEKDAY'])
    features = pd.get_dummies(features, columns=['Status'])
    # features = pd.get_dummies(features, columns=['RD'])

    logger.info('---------------------------------------------')
    logger.info("Data-Preparation finished ...")
    logger.info("Summary statistics after standardization:")
    logger.info(features.describe())
    logger.info(features.info())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=RANDOM_SEED)

    del data_sample, target
    gc.collect()

    logger.info('---------------------------------------------')
    logger.info("Training the model ...")

    if tuning:
        logger.info("You chose to tune the model. This will take a moment...")
        # Use Optuna to tune the hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: bayesian_optimization_forest(
            trial, X_train, y_train), n_trials=20, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")
        best_model = RandomForestClassifier(
            **best_params, random_state=RANDOM_SEED, n_jobs=1, verbose=2)

    ########################################################

    if not tuning:
        best_model = RandomForestClassifier(n_estimators=50,
                                            min_samples_leaf=8,
                                            min_samples_split=10,
                                            max_features=0.5,
                                            max_depth=60,  # mein PC schmiert sonst ab
                                            # max_leaf_nodes=300,
                                            random_state=RANDOM_SEED,
                                            verbose=2,
                                            bootstrap=False,
                                            oob_score=False,
                                            class_weight=None,
                                            n_jobs=2)

    # Modell trainieren
    best_model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    # Modellbewertung
    test_score = best_model.score(X_test, y_test)
    logger.info(f"Test-Set Score: {test_score}")

    logger.info('---------------------------------------------')

    logger.info("Predicting the model ...")
    y_pred = best_model.predict(X_test)
    y_pred_prob_rf = best_model.predict_proba(X_test)

    logger.info('---------------------------------------------')
    classification_rep = classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{classification_rep}")

    feature_importances = pd.DataFrame(best_model.feature_importances_,
                                       index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    logger.info(f"Feature Importances:\n {feature_importances}")

    logger.success(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    unique_classes = np.unique(y_test)

    log_loss_rf = log_loss(y_test, y_pred_prob_rf, labels=unique_classes)

    logger.success(f"Random Forest Log Loss: {log_loss_rf}")
