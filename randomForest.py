import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from loguru import logger
from lib.data_prep import *
import random
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from lib.tune_random_forest import grid_tune_hyperparameters, randomize_tune_hyperparameters
import gc

# RANDOM_SEED = random.randint(1, 10)  # können wir final setten zum Schluss
RANDOM_SEED = 41
if __name__ == "__main__":
    data = load_data()

    print(data.head())
    print(data.info())

    data_sample = data.sample(n=500000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)
    data_sample = remove_outside_la(data_sample)

    del data
    gc.collect()

    features: pd.DataFrame = data_sample[['AREA',
                                          'TIME.OCC',
                                          'Latitude',
                                          'Longitude',
                                          'SEASON',
                                          'WEEKDAY',
                                          'DATE.OCC.Year',
                                          'Diff between OCC and Report',
                                          'Status']]

    features = optimize_data_types(features)

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
        features, target, test_size=0.4, random_state=RANDOM_SEED)

    del data_sample, target
    gc.collect()

    logger.info('---------------------------------------------')
    logger.info("Training the model ...")

    # Spielerrei mit tuning, kann man vielleicht auch löschen

    #############################################################
    # param_grid = {
    #     'n_estimators': [100, 150],
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [10, 20, None],
    #     'min_samples_leaf': [2, 5, None],
    # }

    # param_distributions = {
    #     'n_estimators': [100, 150, 200, 250],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [10, 20, 30],
    #     'min_samples_leaf': [1, 2, 3],
    # }

    # rf_model = RandomForestClassifier(random_state=RANDOM_SEED)

    # # grid_search = tune_hyperparameters(
    # #     rf_model, X_train, y_train, param_grid, random_state=RANDOM_SEED)

    # grid_search = randomize_tune_hyperparameters(
    #     model=rf_model, X_train=X_train, y_train=y_train,
    #     param_distributions=param_distributions, cv_splits=5, n_iter=10,
    #     random_state=42, verbose=2
    # )

    # # Beste Parameter und beste Score anzeigen
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    # logger.info(f"Beste Parameter: {best_params}")
    # logger.info(f"Beste Score: {best_score}")

    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    # logger.info(f"Beste Parameter: {best_params}")
    # logger.info(f"Beste Score: {best_score}")

    # # Beste Modellkonfiguration verwenden
    # best_model = grid_search.best_estimator_

    ########################################################

    # vorher cv gemacht und wir verwenden einfach die Ergebnisse davon

    # Modell mit den besten Parametern initialisieren
    best_model = GradientBoostingClassifier(
        n_estimators=100, max_leaf_nodes=200, random_state=RANDOM_SEED, verbose=2)

    #######################################################

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

    logger.info('---------------------------------------------')
    classification_rep = classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{classification_rep}")

    feature_importances = pd.DataFrame(best_model.feature_importances_,
                                       index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    logger.info(f"Feature Importances:\n {feature_importances}")
    logger.success(f"Accuracy: {accuracy_score(y_test, y_pred)}")
