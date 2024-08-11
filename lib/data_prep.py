import os
import pandas as pd
import zipfile
from io import BytesIO
import numpy as np
from loguru import logger
import re


def load_data() -> pd.DataFrame:

    current_dir = os.getcwd()

    zip_path = os.path.join(
        current_dir, "Data/Crimes_2012-2016.csv.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        csv_file_name = zip_ref.namelist()[0]

        with zip_ref.open(csv_file_name) as csv_file:

            data = BytesIO(csv_file.read())

            df = pd.read_csv(data, sep=",", parse_dates=[
                             "Date.Rptd", "DATE.OCC"])

            df = df.drop_duplicates()

            df['CrmCd.Desc'] = df['CrmCd.Desc'].replace(
                "nan", np.nan)

            df = df.dropna(subset=['CrmCd.Desc'])
            return df


def load_data_train() -> pd.DataFrame:

    current_dir = os.getcwd()

    zip_path = os.path.join(current_dir, "Data/train_data.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        csv_file_name = zip_ref.namelist()[0]

        with zip_ref.open(csv_file_name) as csv_file:

            data = BytesIO(csv_file.read())

            df = pd.read_csv(data, sep=",", parse_dates=[
                             "Date.Rptd", "DATE.OCC"])

            df = df.drop_duplicates()

            df['CrmCd.Desc'] = df['CrmCd.Desc'].replace(
                "nan", np.nan)

            df = df.dropna(subset=['CrmCd.Desc'])
            return df


def load_data_test() -> pd.DataFrame:

    current_dir = os.getcwd()

    zip_path = os.path.join(current_dir, "Data/test_data.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        csv_file_name = zip_ref.namelist()[0]

        with zip_ref.open(csv_file_name) as csv_file:

            data = BytesIO(csv_file.read())

            df = pd.read_csv(data, sep=",", parse_dates=[
                             "Date.Rptd", "DATE.OCC"])

            df = df.drop_duplicates()

            df['CrmCd.Desc'] = df['CrmCd.Desc'].replace(
                "nan", np.nan)

            df = df.dropna(subset=['CrmCd.Desc'])
            return df


def format_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    data['DATE.OCC.Year'] = data['DATE.OCC'].dt.year.astype(int)
    data['DATE.OCC.Month'] = data['DATE.OCC'].dt.month.astype(int)
    data['DATE.OCC.Day'] = data['DATE.OCC'].dt.day.astype(int)

    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
        r'\(([^,]+), ([^)]+)\)').astype('float32')

    data['SEASON'] = data['DATE.OCC.Month'].apply(get_season)
    data['WEEKDAY'] = data['DATE.OCC'].dt.day_name()

    data['Diff between OCC and Report'] = (
        data['Date.Rptd'] - data['DATE.OCC']).dt.days
    data['day_of_month'] = data['DATE.OCC'].dt.day

    data["Street Category"] = data["LOCATION"].apply(
        extract_street_category)
    data = clean_coordinates(data)
    return data


def clean_coordinates(data: pd.DataFrame) -> pd.DataFrame:

    invalid_coords: pd.Series[bool] = (
        data['Latitude'] == 0.0) & (data['Longitude'] == 0.0)
    data.loc[invalid_coords, ['Latitude', 'Longitude']] = [np.nan, np.nan]

    area_coords_mean: pd.DataFrame = data.groupby(
        'AREA')[['Latitude', 'Longitude']].transform('mean')

    data[['Latitude', 'Longitude']] = data[[
        'Latitude', 'Longitude']].fillna(area_coords_mean)

    invalid_coords_after = data['Latitude'].isna() | data['Longitude'].isna()

    count_invalid_coords = invalid_coords_after.sum()
    print(f"Anzahl der Zeilen, die gelöscht werden: {count_invalid_coords}")

    data = data.dropna(subset=['Latitude', 'Longitude'])
    logger.info("Cleaned the coordinates")
    return data


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Frühling'
    elif month in [6, 7, 8]:
        return 'Sommer'
    else:
        return 'Herbst'


# def categorize_time(hour: int) -> str:
#     # Zeit als vierstellige Zeichenkette formatieren
#     hour_str = str(hour).zfill(4)
#     # Die ersten beiden Ziffern der Zeit extrahieren
#     first_two_digits = hour_str[:2]
#     return first_two_digits


def filter_outside_points(df: pd.DataFrame) -> pd.DataFrame:
    # Definieren der Grenzen
    north_bound = 34.337314
    east_bound = -118.155348
    south_bound = 33.704599
    west_bound = -118.668225

    # Filtern der Punkte außerhalb des gewünschten Bereichs
    outside_points = df[
        (df['Latitude'] > north_bound) | (df['Latitude'] < south_bound) |
        (df['Longitude'] > east_bound) | (df['Longitude'] < west_bound)
    ]

    # Zählen der Punkte außerhalb des Bereichs
    count_outside_points = outside_points.shape[0]
    print(f"Anzahl der Punkte außerhalb des gewünschten Bereichs: {
          count_outside_points}")

    # Entfernen der Punkte außerhalb des Bereichs
    data = df.drop(outside_points.index)

    return data


def checking_missing_coordinates(data: pd.DataFrame) -> bool:
    # Überprüft, ob es fehlende Werte in den Spalten 'Latitude' oder 'Longitude' gibt
    missing_latitude = data['Latitude'].isnull().any()
    missing_longitude = data['Longitude'].isnull().any()

    # Gibt True zurück, wenn es fehlende Werte in einer der beiden Spalten gibt
    return missing_latitude or missing_longitude


def extract_street_category(address):
    match = re.search(r'( [A-Z]{2})$', address)
    if match:
        return match.group(1)
    return None


if __name__ == "__main__":

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

    print("Inhalt des aktuellen Verzeichnisses:", os.listdir())

    print("Inhalt des übergeordneten Verzeichnisses:", os.listdir('..'))

    data = load_data_train()

    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
        r'\(([^,]+), ([^)]+)\)').astype(float)

    data = format_data_frame(data)

    print(data.head())

    print(data.info())

    print('Fehlende Werte ?')

    print(checking_missing_coordinates(data))

    format_data_frame(data)

    print('Fehlende Werte ?')

    print(checking_missing_coordinates(data))

    unique_street_category: int = data["Street Category"].nunique()

    print(f"Straßen Arten : {unique_street_category}")

    print(data["Street Category"].unique())
