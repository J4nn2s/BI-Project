from lib.data_prep import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os
from loguru import logger
from tabulate import tabulate


def create_time_series_plot(df: pd.DataFrame, display: bool = True) -> None:
    if df.empty:
        logger.info("DataFrame is empty.")
        return

    try:
        df['Date.Rptd'] = pd.to_datetime(df['Date.Rptd'])
        daily_counts = df.groupby('Date.Rptd').size().reset_index(name='Count')

        sns.set_style('darkgrid')
        sns.set_palette("deep")

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(daily_counts['Date.Rptd'],
                daily_counts['Count'], linewidth=2, color='#1E88E5')
        ax.fill_between(
            daily_counts['Date.Rptd'], daily_counts['Count'], alpha=0.3, color='#1E88E5')

        ax.set_title('Daily Count of Reported Crimes', fontsize=20, pad=20)
        ax.set_xlabel('Date', fontsize=14, labelpad=10)
        ax.set_ylabel('Number of Reported Crimes', fontsize=14, labelpad=10)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))

        plt.gcf().autofmt_xdate()
        ax.grid(True, linestyle='--', alpha=0.7)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        stats_text = (
            f"Total Reports: {daily_counts['Count'].sum():,}\n"
            f"Average Daily Reports: {daily_counts['Count'].mean():.2f}\n"
            f"Max Daily Reports: {daily_counts['Count'].max()}"
        )

        plt.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        os.makedirs('Plots', exist_ok=True)
        plt.savefig('Plots/time_series_plot.png')
        logger.info("Plot saved as 'Plots/time_series_plot.png'")

    except Exception as e:
        logger.info(f"An error occurred: {e}")


def plot_area_code_frequencies(df: pd.DataFrame) -> None:
    area_counts = df['AREA'].value_counts().reset_index()
    area_counts.columns = ['Area', 'Frequency']

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Area', y='Frequency', hue='Area',
                data=area_counts, palette='deep', dodge=False, legend=False)

    plt.title('Frequency of Each Area Code', fontsize=20)
    plt.xlabel('Area Code', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45)

    plt.tight_layout()

    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/area_code_frequencies.png')
    logger.info("Plot saved as 'Plots/area_code_frequencies.png'")


def histogram_time(data: pd.DataFrame) -> None:
    data['TIME.OCC'] = data['TIME.OCC'].astype(str).str.zfill(4)

    # Extract hour and minute using vectorized string operations
    data['Hour'] = data['TIME.OCC'].str[:2].astype(int)
    data['Minute'] = data['TIME.OCC'].str[2:].astype(int)

    # Plotting histogram of hours
    plt.figure(figsize=(10, 6))
    plt.hist(data['Hour'], bins=24, edgecolor='k', alpha=0.7)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency')
    plt.title('Distribution of Times in the Day')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/histogram_time_dist.png')
    logger.info("Plot saved as 'Plots/histogram_time_dist.png'")


def plot_top10_crimes(data: pd.DataFrame) -> None:
    common_crimes = ['TRAFFIC DR', 'BATTERY - SIMPLE ASSAULT',
                     'VEHICLE - STOLEN', 'BURGLARY FROM VEHICLE', 'BURGLARY',
                     'THEFT PLAIN - PETTY ($950 & UNDER)',
                     'THEFT OF IDENTITY', 'SPOUSAL(COHAB) ABUSE - SIMPLE ASSAULT',
                     'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
                     'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)']
    df_filtered = data[data['CrmCd.Desc'].isin(common_crimes)].copy()

    df_filtered.loc[:, 'TIME.OCC'] = df_filtered['TIME.OCC'].astype(
        str).str.zfill(4)

    # Extract hour using vectorized string operations
    df_filtered.loc[:, 'Hour'] = df_filtered['TIME.OCC'].str[:2].astype(int)

    # Aggregate data by crime type and hour
    aggregated_data = df_filtered.groupby(
        ['CrmCd.Desc', 'Hour']).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 10))

    for i, crime in enumerate(common_crimes, 1):
        plt.subplot(5, 2, i)
        if crime in aggregated_data.index:
            crime_data = aggregated_data.loc[crime]
            plt.plot(crime_data.index, crime_data.values, marker='o')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Frequency')
            plt.title(f'Time Series for {crime}')
            plt.xticks(range(0, 24))
            plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/top10_during_the_day.png')
    logger.info("Plot saved as 'Plots/top10_during_the_day.png'")


def get_count_of_crimes(df: pd.DataFrame) -> pd.Series:
    top_10_crmcd_desc = df['CrmCd.Desc'].value_counts().head(10)
    return top_10_crmcd_desc


def get_count_of_areas(df: pd.DataFrame) -> pd.Series:
    area_count = df['AREA'].value_counts()
    return area_count


def checking_missing_values(data: pd.DataFrame) -> bool:
    missing = data.isnull().values.any()
    return missing


def counting_missing_values(data: pd.DataFrame) -> int:
    return data.isnull().sum().sum()


def columns_with_missing_values(data: pd.DataFrame) -> list:
    return [col for col in data.columns if data[col].isnull().any()]


def counting_missing_values(data: pd.DataFrame) -> int:
    return data.isnull().sum().sum()


def counting_missing_values_column(data: pd.DataFrame) -> list[int]:
    return data.isnull().sum()

def count_outside_la(data: pd.DataFrame) -> int:
    lat_min, lat_max = 33.0, 36.0
    long_min, long_max = -120.0, -116.0

    data_not_in_la = data[~((data['Latitude'] >= lat_min) & (data['Latitude'] <= lat_max) &
                            (data['Longitude'] >= long_min) & (data['Longitude'] <= long_max))]

    # Berechnen der Anzahl der Zeilen, die nicht in LA sind
    return len(data_not_in_la)

if __name__ == "__main__":

    logger.info("Load Data ...")
    data: pd.DataFrame = load_data()

    missing: bool = checking_missing_values(data)
    if missing:
        logger.warning("There are missing values inside the Dataframe")
        total_missing_count: int = counting_missing_values(data)
        logger.warning(
            f"\nTotal number of missing values in the DataFrame:, {total_missing_count}")
        missing_columns = columns_with_missing_values(data)
        logger.warning(f"Columns with missing values:, {missing_columns}")
        # Display a sample of rows with missing values
        logger.info(data[data.isnull().any(axis=1)].head())
        missing_values_count_column = counting_missing_values_column(data)
        logger.warning(f"Missing values count by column: {
            missing_values_count_column}")
    else:
        logger.info("No missing values inside the Dataframe")

    logger.info("Data loaded: First Information About the Dataframe")

    logger.info("Datatypes of the Script")
    logger.info(data.info())

    logger.info("Checking the first rows")
    logger.info(data.head())

    logger.info("Looking into the distribution of the variables")

    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(r'\(([^,]+),\s*([^\)]+)\)')
    # Konvertiere die neuen Spalten in numerische Werte
    data['Latitude'] = pd.to_numeric(data['Latitude'])
    data['Longitude'] = pd.to_numeric(data['Longitude'])

    describe_table = data.describe().reset_index()
    logger.info(tabulate(describe_table, headers='keys', tablefmt='pretty'))

    logger.info("Checking whether Coordinates are correct")
    logger.info(f" Count of data outside of LA: {count_outside_la(data)}")
    ########################### THE PLOTS ##########################

    # create_time_series_plot(data)
    # plot_area_code_frequencies(data)
    # histogram_time(data)
    # plot_top10_crimes(data)
    logger.info(get_count_of_crimes(data))
    num_unique_areas = data['AREA'].nunique()
    logger.info(f"Count of different 'AREA': {num_unique_areas}")
    logger.info(f"Count per 'AREA': {get_count_of_areas(data)}")
    num_unique_crime_categories = data["Crm.Cd"].nunique()
    logger.info(f"Count of different 'Crm.Cd': {num_unique_crime_categories}")
