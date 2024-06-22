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
    data['DATE.OCC.Year'] = data['DATE.OCC'].dt.year
    data['DATE.OCC.Month'] = data['DATE.OCC'].dt.month
    data['DATE.OCC.Day'] = data['DATE.OCC'].dt.day

    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
        r'\(([^,]+), ([^)]+)\)').astype(float)
    print(data.describe())
    return data


if __name__ == "__main__":

    import os

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())
    print("Inhalt des aktuellen Verzeichnisses:", os.listdir())
    print("Inhalt des übergeordneten Verzeichnisses:", os.listdir('..'))

    data = load_data()
    print(data.head())
    print(data.info())
    format_data_frame(data)
