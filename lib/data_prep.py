import os
import pandas as pd

def load_data() -> pd.DataFrame:
    parent_directory = os.path.dirname(os.getcwd())

    file_path = os.path.join(parent_directory, "Crimes_2012-2016.csv")
    return pd.read_csv(file_path, sep=",", parse_dates=["Date.Rptd", "DATE.OCC"])

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    pass

if __name__ == "__main__":
    load_data()