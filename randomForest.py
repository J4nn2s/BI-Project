import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from loguru import logger
from lib.data_prep import load_data, format_data_frame
import random

RANDOM_SEED = random.randint(1, 10)  # können wir final setten zum Schluss

if __name__ == "__main__":
    data = load_data()

    print(data.head())
    print(data.info())

    data_sample = data.sample(n=50000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)

    features = data_sample[['TIME.OCC', 'AREA', 'DATE.OCC.Year',
                            'DATE.OCC.Month', 'DATE.OCC.Day', 'Latitude', 'Longitude']]
    target = data_sample['Crm.Cd']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features[['Latitude', 'Longitude', 'TIME.OCC']])

    features.loc[:, ['Latitude', 'Longitude', 'TIME.OCC']] = scaled_features

    logger.info('---------------------------------------------')
    logger.info("Data-Preparation finished ...")
    logger.info("Summary statistics after standardization:")
    logger.info(features.describe())
    logger.info(features.head())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_SEED)

    logger.info('---------------------------------------------')
    logger.info("Training the model ...")

    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_SEED, criterion='gini')
    rf_model.fit(X_train, y_train)

    logger.info('---------------------------------------------')

    logger.info("Predicting the model ...")
    y_pred = rf_model.predict(X_test)

    logger.info('---------------------------------------------')
    logger.info(f"\nClassification Report:\n{
        classification_report(y_test, y_pred)}")

    feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                       index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    logger.success(f"Accuracy: {accuracy_score(y_test, y_pred)}")
