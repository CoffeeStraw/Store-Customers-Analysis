"""
Data Mining Project
Authors: CoffeeStraw, sd3ntato
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fix_dataset(df: pd.DataFrame):
    """
    Performs some data quality operations on the dataset
    """
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Converts sale to float
    df["Sale"] = df["Sale"].str.replace(',', '.').astype(float)

    # Converts CustomerID to int, replacing nulls with -1
    df["CustomerID"] = df["CustomerID"].fillna(-1).astype(int)

    df.to_csv("customer_supermarket_2.csv")


def statistics_basketID(df: pd.DataFrame):
    """Collect statistics for basketID attribute.
    """
    basketID_freq = df["BasketID"].value_counts()
    mean, median, mode = basketID_freq.mean(), basketID_freq.median(), basketID_freq.mode()[0]
    print(mean, median, mode)


def statistics_basketDate(df: pd.DataFrame):
    """Draw various plots based on the initial status of the dataset
    """
    days_freq = np.array(df["BasketDate"].dt.date)
    days, freq = np.unique(days_freq, return_counts=True)

    plt.rcParams['figure.figsize'] = 15, 6
    fig, ax = plt.subplots()
    ax.plot(days, freq)

    ax.set(xlabel='Days', ylabel='Frequency', title='Frequency of purchases in each day of the dataset.')
    ax.grid()

    plt.tight_layout()
    fig.savefig("BasketDatePlot.png")
    plt.show()


if __name__ == "__main__":
    """
    df = pd.read_csv('customer_supermarket.csv', sep='\t', index_col=0)

    # Prints data's samples and informations,
    # including the number of not null values for each columns
    print(df.info(), "\n")
    print(df.head())

    fix_dataset(df)
    quit()
    """

    df = pd.read_csv('customer_supermarket.csv', sep='\t', index_col=0, parse_dates=["BasketDate"])
    
    # === Calculate statistical informations ===
    """
    statistics_basketID(df)
    statistics_basketDate(df)
    """

    # print(df['CustomerCountry'].unique())
    # print(df[df['CustomerCountry'].str.contains("Unspecified")].to_string())

    # print(df['Qta'].unique())
