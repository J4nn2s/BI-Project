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
        lambda x: str(x).zfill(4)[:2]).astype(int)

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

def remove_outside_la(data: pd.DataFrame) -> pd.DataFrame:
    lat_min, lat_max = 33.0, 36.0
    long_min, long_max = -120.0, -116.0

    # Filtere die Zeilen, die innerhalb des Bereichs liegen
    data_in_la = data[(data['Latitude'] >= lat_min) & (data['Latitude'] <= lat_max) &
                      (data['Longitude'] >= long_min) & (data['Longitude'] <= long_max)]

    # Rückgabe des DataFrame, der nur Einträge innerhalb des angegebenen Bereichs enthält
    return data_in_la


def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            if col_type in ['int64', 'int32']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type in ['float64', 'float32']:
                df[col] = pd.to_numeric(df[col], downcast='float')
    return df


if __name__ == "__main__":

    import os

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())
    print("Inhalt des aktuellen Verzeichnisses:", os.listdir())
    print("Inhalt des übergeordneten Verzeichnisses:", os.listdir('..'))

    data = load_data()
    print(data.head())
    print(data.info())
    format_data_frame(data)
