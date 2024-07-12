import os
import pandas as pd
import zipfile
from io import BytesIO
import numpy as np
from loguru import logger


def load_data() -> pd.DataFrame:
    current_dir = os.getcwd()
    zip_path = os.path.join(current_dir, "Data/Crimes_2012-2016.csv.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Annahme: Es gibt nur eine Datei in der ZIP
        csv_file_name = zip_ref.namelist()[0]
        with zip_ref.open(csv_file_name) as csv_file:
            # BytesIO wird verwendet, um die Datei im Speicher zu halten
            data = BytesIO(csv_file.read())
            df = pd.read_csv(data, sep=",", parse_dates=[
                             "Date.Rptd", "DATE.OCC"])

            df = df.drop_duplicates()

            df['CrmCd.Desc'] = df['CrmCd.Desc'].replace(
                "nan", np.nan)

            df = df.dropna(subset=['CrmCd.Desc'])
            return df


def load_data_less_memory() -> pd.DataFrame:
    dtype_dict = {
        'DR.NO': 'int32',
        'TIME.OCC': 'int32',
        'AREA': 'int32',
        'AREA.NAME': 'category',
        'RD': 'int32',
        'Crm.Cd': 'int32',
        'CrmCd.Desc': 'category',
        'Status': 'category',
        'Status.Desc': 'category',
        'LOCATION': 'category',
        'Cross.Street': 'category',
        'Location.1': 'category'
    }
    current_dir = os.getcwd()
    zip_path = os.path.join(current_dir, "Data/Crimes_2012-2016.csv.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Annahme: Es gibt nur eine Datei in der ZIP
        csv_file_name = zip_ref.namelist()[0]
        with zip_ref.open(csv_file_name) as csv_file:
            # BytesIO wird verwendet, um die Datei im Speicher zu halten
            data = BytesIO(csv_file.read())
            df = pd.read_csv(data, dtype=dtype_dict, sep=",", parse_dates=[
                             "Date.Rptd", "DATE.OCC"])

            df = df.drop_duplicates()
            df = df[df["CrmCd.Desc"] != "nan"]
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

def remove_outside_la(data: pd.DataFrame) -> pd.DataFrame:
    lat_min, lat_max = 33.0, 36.0
    long_min, long_max = -120.0, -116.0

    # Filtere die Zeilen, die innerhalb des Bereichs liegen
    data_in_la = data[(data['Latitude'] >= lat_min) & (data['Latitude'] <= lat_max) &
                      (data['Longitude'] >= long_min) & (data['Longitude'] <= long_max)]

    # Rückgabe des DataFrame, der nur Einträge innerhalb des angegebenen Bereichs enthält
    return data_in_la


def checking_missing_coordinates(data: pd.DataFrame) -> bool:
    # Überprüft, ob es fehlende Werte in den Spalten 'Latitude' oder 'Longitude' gibt
    missing_latitude = data['Latitude'].isnull().any()
    missing_longitude = data['Longitude'].isnull().any()

    # Gibt True zurück, wenn es fehlende Werte in einer der beiden Spalten gibt
    return missing_latitude or missing_longitude


if __name__ == "__main__":

    import os

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())
    print("Inhalt des aktuellen Verzeichnisses:", os.listdir())
    print("Inhalt des übergeordneten Verzeichnisses:", os.listdir('..'))

    data = load_data_less_memory()
    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
        r'\(([^,]+), ([^)]+)\)').astype(float)

    print(data.head())
    print(data.info())
    print('Fehlende Werte ?')
    print(checking_missing_coordinates(data))

    format_data_frame(data)
    print('Fehlende Werte ?')

    print(checking_missing_coordinates(data))
