import os
import pandas as pd
if __name__ == "__main__":
    parent_directory = os.path.dirname(os.getcwd())

    file_path = os.path.join(parent_directory, "Crimes_2012-2016.csv")
    data = pd.read_csv(file_path, sep=",")
    print(data.info())
