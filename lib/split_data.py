import pandas as pd
from sklearn.model_selection import train_test_split
import os
import zipfile
from io import BytesIO
import numpy as np


def split_data(data, test_size=0.3, output_dir='Data'):
    os.makedirs(output_dir, exist_ok=True)

    X = data.drop(columns=['CrmCd.Desc'])
    y = data['CrmCd.Desc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    train_data = pd.concat([X_train, y_train], axis=1)
    train_csv = train_data.to_csv(index=False)

    test_data = pd.concat([X_test, y_test], axis=1)
    test_csv = test_data.to_csv(index=False)

    # ZIP-Datei für Trainingsdaten erstellen
    train_zip_path = os.path.join(output_dir, 'train_data.csv.zip')
    with zipfile.ZipFile(train_zip_path, 'w') as zipf:
        zipf.writestr('train_data.csv', train_csv)

    # ZIP-Datei für Testdaten erstellen
    test_zip_path = os.path.join(output_dir, 'test_data.csv.zip')
    with zipfile.ZipFile(test_zip_path, 'w') as zipf:
        zipf.writestr('test_data.csv', test_csv)


def load_data_split() -> pd.DataFrame:
    current_dir = os.getcwd()
    zip_path = os.path.join(current_dir, "Data/Crimes_2012-2016.csv.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_file_name = zip_ref.namelist()[0]

        with zip_ref.open(csv_file_name) as csv_file:

            data = BytesIO(csv_file.read())
            df = pd.read_csv(data, sep=",", parse_dates=[
                             "Date.Rptd", "DATE.OCC"])

    return df


if __name__ == "__main__":
    data: pd.DataFrame = load_data_split()
    split_data(data)
