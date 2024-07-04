import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

    data_sample = data.sample(n=300000, random_state=RANDOM_SEED)
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

    best_model = DecisionTreeClassifier(
        max_leaf_nodes=200, random_state=RANDOM_SEED)

    #######################################################

    # Modell trainieren
    best_model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    y_pred = best_model.predict(X_test)
    # Wahrscheinlichkeiten für die positive Klasse
    y_prob = best_model.predict_proba(X_test)[:, 1]

    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Konfusionsmatrix
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)

    # Ausgabe der Accuracy
    logger.success(f"Accuracy: {accuracy:.4f}")  # Erfolgsnachricht
