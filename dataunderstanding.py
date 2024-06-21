from lib.data_prep import load_data
import pandas as pd

if __name__ == "__main__":
    print("Load Data ...")
    data: pd.DataFrame = load_data()

    print("Data loaded: First Information About the Dataframe")

    print(data.info())
    print(data.head())