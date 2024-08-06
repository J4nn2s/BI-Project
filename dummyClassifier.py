from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from lib.data_prep import *
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from lib.crimeCategories import crime_categories, categorize_crime


class CustomDummyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):

        self.most_frequent_ = defaultdict(lambda: y.mode()[0])

        for area in X['AREA'].unique():
            for hour in X['HOUR'].unique():
                mask = (X['AREA'] == area) & (X['HOUR'] == hour)
                if mask.any():
                    self.most_frequent_[(area, hour)] = y[mask].mode()[0]
        return self

    def predict(self, X):
        return X.apply(lambda row: self.most_frequent_[(row['AREA'], row['HOUR'])], axis=1)


if __name__ == "__main__":
    RANDOM_SEED = 42
    data_sample = load_data()

    # print(data.head())
    # print(data.info())

    # data_sample = data.sample(n=500000, random_state=RANDOM_SEED)
    data_sample = format_data_frame(data_sample)
    data_sample = filter_outside_points(data_sample)
    logger.info(f"Grouping Categories")
    data_sample['Crime Categorie'] = data_sample['CrmCd.Desc'].apply(
        categorize_crime)

    target = data_sample['Crime Categorie']
    features: pd.DataFrame = data_sample.drop(columns=["Crime Categorie"])
    features["HOUR"] = features["TIME.OCC"].astype(str).str[0:2]

    logger.info('---------------------------------------------')
    logger.info("Data-Preparation finished ...")
    logger.info("Summary statistics after standardization:")
    logger.info(features.describe())
    logger.info(features.info())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=RANDOM_SEED)

    logger.info("Training the DummyClassifier")

    dummy_clf = CustomDummyClassifier()
    dummy_clf.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred_dummy = dummy_clf.predict(X_test)

    # y_pred_proba_dummy = dummy_clf.predict_proba(X_test)

    # Genauigkeit berechnen
    accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
    logger.success(f"Dummy Classifier Accuracy: {accuracy_dummy}")

    # log_loss berechnen
    # log_loss_dummy = log_loss(y_test, y_pred_proba_dummy)
    # logger.success(f"Dummy Classifier Log Loss: {log_loss_dummy}")


'''
:<module>:50 - Training the DummyClassifier
:<module>:60 - Dummy Classifier Accuracy: 0.16814607384707508
'''
