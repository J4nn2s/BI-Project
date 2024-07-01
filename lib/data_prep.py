import os
import pandas as pd
import zipfile
from io import BytesIO


def load_data() -> pd.DataFrame:
    current_dir = os.getcwd()
    zip_path = os.path.join(current_dir, "Data/Crimes_2012-2016.csv.zip")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Annahme: Es gibt nur eine Datei in der ZIP
        csv_file_name = zip_ref.namelist()[0]
        with zip_ref.open(csv_file_name) as csv_file:
            # BytesIO wird verwendet, um die Datei im Speicher zu halten
            data = BytesIO(csv_file.read())
            return pd.read_csv(data, sep=",", parse_dates=["Date.Rptd", "DATE.OCC"])


def format_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    data['DATE.OCC.Year'] = data['DATE.OCC'].dt.year.astype(int)
    data['DATE.OCC.Month'] = data['DATE.OCC'].dt.month.astype(int)
    data['DATE.OCC.Day'] = data['DATE.OCC'].dt.day.astype(int)

    # Stellen sicher, dass es vierstellig ist
    data['TIME_CATEGORY'] = data['TIME.OCC'].apply(
        lambda x: str(x).zfill(4)[:2])

    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
        r'\(([^,]+), ([^)]+)\)').astype(float)
    data['SEASON'] = data['DATE.OCC.Month'].apply(get_season)
    data['WEEKDAY'] = data['DATE.OCC'].dt.day_name()
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

if __name__ == "__main__":

    import os

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())
    print("Inhalt des aktuellen Verzeichnisses:", os.listdir())
    print("Inhalt des übergeordneten Verzeichnisses:", os.listdir('..'))

    data = load_data()
    print(data.head())
    print(data.info())
    format_data_frame(data)
