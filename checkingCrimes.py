import os
import pandas as pd
from lib.data_prep import load_data
from loguru import logger


if __name__ == "__main__":
    logger.info("Load Data ...")

    input_file_path = os.path.join('Data', 'unique_values.txt')

    data = load_data()
    category_counts = data["CrmCd.Desc"].value_counts()

    categories_less_than_50 = category_counts[category_counts < 100]

    num_categories_less_than_50 = len(categories_less_than_50)
    logger.info(f"Number of 'CrmCd.Desc' categories with less than 50 observations: {
                num_categories_less_than_50}")

    logger.info(f" hier die Kategorien: {categories_less_than_50}")
    categories_less_than_50 = category_counts[category_counts < 100].index

    rows_less_than_100 = data[data['CrmCd.Desc'].isin(
        categories_less_than_50)]

    num_rows_less_than_100 = len(rows_less_than_100)
    logger.info(f"Number of rows in categories with less than 100 observations: {
                num_rows_less_than_100}")

    unique_values = data['CrmCd.Desc'].unique()

    os.makedirs('Data', exist_ok=True)

    file_path = os.path.join('Data', 'unique_values.txt')
    with open(file_path, 'w') as f:
        for value in unique_values:
            f.write(f"{value}\n")

    logger.info(f"Unique values have been saved to '{file_path}'")

    # Save unique values to a text file
    with open('unique_values.txt', 'w') as f:
        for value in unique_values:
            f.write(f"{value}\n")
