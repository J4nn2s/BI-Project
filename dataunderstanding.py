from lib.data_prep import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os


def create_time_series_plot(df: pd.DataFrame, display: bool = True) -> None:
    if df.empty:
        print("DataFrame is empty.")
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
        print("Plot saved as 'Plots/time_series_plot.png'")

    except Exception as e:
        print(f"An error occurred: {e}")


def save_and_show_image():
    try:
        from PIL import Image
        img = Image.open('Plots/time_series_plot.png')
        img.show()
    except Exception as e:
        print(f"An error occurred while opening the saved image: {e}")


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
    print("Plot saved as 'Plots/area_code_frequencies.png'")


def get_top_10_crmcd_desc(df: pd.DataFrame) -> pd.Series:
    top_10_crmcd_desc = df['CrmCd.Desc'].value_counts().head(10)
    return top_10_crmcd_desc


if __name__ == "__main__":
    print("Load Data ...")
    data: pd.DataFrame = load_data()

    print("Data loaded: First Information About the Dataframe")

    "Datatypes of the Script"
    print(data.info())

    "Checking the first rows"
    print(data.head())

    # create_time_series_plot(data)
    # plot_area_code_frequencies(data)
    print(get_top_10_crmcd_desc(data))
