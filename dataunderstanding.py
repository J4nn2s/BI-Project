from lib.data_prep import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os
from loguru import logger
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


def clean_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
        r'\(([^,]+), ([^)]+)\)').astype(float)

    invalid_coords = (data['Latitude'] == 0.0) & (data['Longitude'] == 0.0)
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
    df_filtered = df.drop(outside_points.index)

    return df_filtered


def plot_crime_heatmap2(df: pd.DataFrame) -> None:
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        logger.error(
            "Columns 'Latitude' and 'Longitude' not found in DataFrame")
        return

    df = df.dropna(subset=['Latitude', 'Longitude'])

    if df.empty:
        logger.warning(
            "No data available after dropping rows with missing 'Latitude' or 'Longitude'")
        return

    # Create a base map
    base_map = folium.Map(
        location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    # Create a heatmap layer
    heat_data = [[row['Latitude'], row['Longitude']]
                 for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(base_map)

    # Save the map as an HTML file
    map_path = 'Plots/crime_heatmap_filtered_map.html'
    os.makedirs('Plots', exist_ok=True)
    base_map.save(map_path)
    logger.info(f"Crime heatmap saved as {map_path}")


def plot_crime_heatmap(df: pd.DataFrame) -> None:
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        logger.error(
            "Columns 'Latitude' and 'Longitude' not found in DataFrame")
        return

    df = df.dropna(subset=['Latitude', 'Longitude'])

    if df.empty:
        logger.warning(
            "No data available after dropping rows with missing 'Latitude' or 'Longitude'")
        return

    # Create a base map
    base_map = folium.Map(
        location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    # Create a heatmap layer
    heat_data = [[row['Latitude'], row['Longitude']]
                 for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(base_map)

    # Save the map as an HTML file
    map_path = 'Plots/crime_heatmap_map.html'
    os.makedirs('Plots', exist_ok=True)
    base_map.save(map_path)
    logger.info(f"Crime heatmap saved as {map_path}")


def detailed_yearly_comparison(df: pd.DataFrame, year: int) -> None:
    df['Date.Rptd'] = pd.to_datetime(df['Date.Rptd'])
    df['Year'] = df['Date.Rptd'].dt.year

    # Filter data for the specified year and the previous yearW
    year_data = df[df['Year'] == year]
    prev_year_data = df[df['Year'] == year - 1]

    monthly_counts_year = year_data.groupby(
        year_data['Date.Rptd'].dt.month).size()
    monthly_counts_prev_year = prev_year_data.groupby(
        prev_year_data['Date.Rptd'].dt.month).size()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_counts_year.index, monthly_counts_year.values,
             marker='o', label=f'{year}')
    plt.plot(monthly_counts_prev_year.index,
             monthly_counts_prev_year.values, marker='o', label=f'{year - 1}')
    plt.title(f'Crime Comparison: {year} vs {year - 1}')
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig(f'Plots/yearly_comparison_{year}.png')
    plt.show()


def plot_long_term_trends(df: pd.DataFrame) -> None:
    if 'Date.Rptd' not in df.columns:
        logger.error("Column 'Date.Rptd' not found in DataFrame")
        return

    if df['Date.Rptd'].isnull().all():
        logger.error("'Date.Rptd' column contains all null values")
        return

    try:
        df['Date.Rptd'] = pd.to_datetime(df['Date.Rptd'])
        df['Year'] = df['Date.Rptd'].dt.year
        yearly_counts = df.groupby('Year').size()

        if yearly_counts.empty:
            logger.warning("Yearly counts are empty")
            return

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o')
        plt.title('Yearly Trends of Crimes')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        os.makedirs('Plots', exist_ok=True)
        plt.savefig('Plots/yearly_trends.png')
        logger.info("Yearly trends saved as 'Plots/yearly_trends.png'")
        plt.show()

    except Exception as e:
        logger.error(f"An error occurred while plotting long term trends: {e}")


def plot_top5_crimes_weekday_analysis(df: pd.DataFrame) -> None:
    # Convert 'Date.Rptd' to datetime and extract 'DayOfWeek'
    df['Date.Rptd'] = pd.to_datetime(df['Date.Rptd'])
    df['DayOfWeek'] = df['Date.Rptd'].dt.dayofweek

    # Get the top 5 crime types
    top5_crimes = df['CrmCd.Desc'].value_counts().head(5).index

    # Filter the DataFrame to include only the top 5 crimes
    df_top5 = df[df['CrmCd.Desc'].isin(top5_crimes)]

    # Group by 'CrmCd.Desc' and 'DayOfWeek' and count the occurrences
    crime_day_counts = df_top5.groupby(
        ['CrmCd.Desc', 'DayOfWeek']).size().reset_index(name='Count')

    # Pivot the data for easier plotting
    crime_day_pivot = crime_day_counts.pivot(
        index='DayOfWeek', columns='CrmCd.Desc', values='Count')

    # Plotting
    plt.figure(figsize=(14, 8))
    crime_day_pivot.plot(kind='bar', stacked=True,
                         figsize=(14, 8), colormap='tab20')

    plt.title('Top 5 Crimes by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Crimes')
    plt.xticks(ticks=range(7), labels=[
               'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/top5_crimes_weekday_analysis.png')
    logger.info(
        "Top 5 crimes weekday analysis saved as 'Plots/top5_crimes_weekday_analysis.png'")
    plt.show()


def plot_seasonal_analysis(df: pd.DataFrame) -> None:
    df['Month'] = df['Date.Rptd'].dt.month
    df['DayOfWeek'] = df['Date.Rptd'].dt.dayofweek

    monthly_counts = df.groupby('Month').size()
    weekly_counts = df.groupby('DayOfWeek').size()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x=monthly_counts.index, y=monthly_counts.values,
                ax=axes[0], palette='deep')
    axes[0].set_title('Crimes per Month')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Number of Crimes')

    sns.barplot(x=weekly_counts.index, y=weekly_counts.values,
                ax=axes[1], palette='deep')
    axes[1].set_title('Crimes per Day of the Week')
    axes[1].set_xlabel('Day of the Week')
    axes[1].set_ylabel('Number of Crimes')
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.tight_layout()
    os.makedirs('Plots', exist_ok=True)
    plt.savefig('Plots/seasonal_analysis.png')
    logger.info("Seasonal analysis saved as 'Plots/seasonal_analysis.png'")


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

# to do Recherche Datenpunkte sauber plotten


# def show_coordinates(df: pd.DataFrame) -> None:
    df = df.dropna(subset=['Latitude', 'Longitude'])

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

    # Los Angeles Koordinaten für Mittelpunkt
    la_center = Point(-118.2437, 34.0522)

    # Karte plotten
    fig, ax = plt.subplots(figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Weltkarte plotten
    world.boundary.plot(ax=ax, linewidth=1)
    gdf.plot(ax=ax, color='red', markersize=5)

    # Mittelpunkt und Zoom auf Los Angeles
    ax.set_xlim([la_center.x - 0.5, la_center.x + 0.5])
    ax.set_ylim([la_center.y - 0.5, la_center.y + 0.5])
    plt.title('Latitude and Longitude Points in Los Angeles')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Erstelle den Pfad zum Speichern des Plots
    plot_directory = 'Plots'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    file_path = os.path.join(plot_directory, 'la_map.png')

    # Plot als PNG-Datei speichern
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

    print(f"Karte wurde als '{file_path}' gespeichert.")


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
    common_crimes = ['TRAFFIC DR #', 'BATTERY - SIMPLE ASSAULT',
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
    # unique_values = data['CrmCd.Desc'].unique()

    # os.makedirs('Data', exist_ok=True)

    # file_path = os.path.join('Data', 'unique_values.txt')
    # with open(file_path, 'w') as f:
    #     for value in unique_values:
    #         f.write(f"{value}\n")

    # logger.info(f"Unique values have been saved to '{file_path}'")

    # # Save unique values to a text file
    # with open('unique_values.txt', 'w') as f:
    #     for value in unique_values:
    #         f.write(f"{value}\n")

    for i in range(2):
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

        data = format_data_frame(data)
        count_invalid_latitude = data['Latitude'].value_counts().get(
            0.0,  np.nan)
        logger.info(f"Anzahl der Zeilen, bei denen Latitude 0.0 oder 0 ist: {
            count_invalid_latitude}")

    logger.info("Data loaded: First Information About the Dataframe")

    logger.info("Datatypes of the Script")
    logger.info(data.info())

    logger.info("Looking into the distribution of the variables")

    logger.info(data.describe())

    logger.info("Checking whether Coordinates are correct")
    logger.info(f" Count of data outside of LA: {count_outside_la(data)}")
    logger.info(get_count_of_crimes(data))

    num_unique_areas = data['AREA'].nunique()

    logger.info(f"Count of different 'AREA': {num_unique_areas}")
    logger.info(f"Count per 'AREA': {get_count_of_areas(data)}")

    num_unique_crime_categories = data["Crm.Cd"].nunique()
    logger.info(f"Count of different 'Crm.Cd': {num_unique_crime_categories}")

    num_unique_crime_describtions = data["CrmCd.Desc"].nunique()
    logger.info(f"Count of different describtions for crime 'CrmCd.Desc': {
                num_unique_crime_categories}")

    num_unique_street = data['Cross.Street'].nunique()
    logger.info(f"Count of different 'street': {num_unique_street}")

    data = clean_coordinates(data)

    logger.info("Data loaded: First Information About the Dataframe")
    logger.info("Datatypes of the DataFrame")
    logger.info(data.info())

    logger.info("Checking the first rows")
    logger.info(data.head())

    outside_points = filter_outside_points(data)
    logger.info("First few rows of outside points")
    logger.info(outside_points.head())

    # Speichern der Datenpunkte außerhalb des gewünschten Bereichs
    # outside_points.to_csv('outside_points.csv', index=False)
    # logger.info("Outside points saved to 'outside_points.csv'")

    data_filtered = filter_outside_points(data)
    logger.info("Filtered data: First Information About the Dataframe")
    logger.info(data_filtered.info())

    data = clean_coordinates(data)

    logger.info("Data loaded: First Information About the Dataframe")
    logger.info("Datatypes of the DataFrame")
    logger.info(data.info())

    logger.info("Checking the first rows")
    logger.info(data.head())

    outside_points = filter_outside_points(data)
    logger.info("First few rows of outside points")
    logger.info(outside_points.head())

    # Speichern der Datenpunkte außerhalb des gewünschten Bereichs
    # outside_points.to_csv('outside_points.csv', index=False)
    logger.info("Outside points saved to 'outside_points.csv'")

    # data_filtered = filter_outside_points(data)
    logger.info("Filtered data: First Information About the Dataframe")
    logger.info(data_filtered.info())

    category_counts = data["Crm.Cd"].value_counts()

    categories_less_than_50 = category_counts[category_counts < 50]

    num_categories_less_than_50 = len(categories_less_than_50)
    logger.info(f"Number of 'Crm.Cd' categories with less than 50 observations: {
                num_categories_less_than_50}")

    # THE PLOTS ##########################W

    # # create_time_series_plot(data)
    # # plot_area_code_frequencies(data)
    # # histogram_time(data)
    # # plot_top10_crimes(data)
    # # show_coordinates(data)
    # # plot_seasonal_analysis(data)
    # # plot_long_term_trends(data)
    # # detailed_yearly_comparison(data, 2016)
    # plot_top5_crimes_weekday_analysis(data)
    # plot_crime_heatmap(data)
    # plot_crime_heatmap2(data_filtered)
