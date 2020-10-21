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

    # Replace nulls in project description with empty strings
    df["ProdDescr"] = df["ProdDescr"].fillna(" ").astype(str)

    # === REMOVE THE OUTLIERS ===

    def get_outliers(s: pd.Series):
        """
        Returns a true-list of the outliers in a column of the DataFrame,
        based on the quantiles
        """
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        trueList = ~((s < (Q1 - 1.5 * IQR)) |(s > (Q3 + 1.5 * IQR)))
        return trueList
    
    df = df[get_outliers(df["Qta"])]
    # df = df[get_outliers(df["Sale"])] # To be discussed

    df.to_csv("customer_supermarket_2.csv")


def customer_profilation(df: pd.DataFrame):
    """Create a new dataset with a profilation of each customer.
    """
    # TODO
    groups = df.groupby(df["CustomerID"])
    pd.Series(data=[sum(group[1]["Qta"]) for group in groups], index=[group[0] for group in groups])


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


def statistics_basketDate2(df: pd.DataFrame):
    # date_freq = df["BasketDate"].dt.date.value_counts() 
    # date_freq.plot()

    print(df["BasketDate"].value_counts())

    g = df["BasketDate"].value_counts().groupby( lambda x : (x.year, x.month) )

    b = pd.DataFrame([group.to_numpy() for _, group in g]).transpose()
    labels = [f"{tmp[0]}/{tmp[1]}" for tmp in g.groups.keys()]
    b.columns = labels
    b.boxplot(rot=-30)
    plt.show()


def statistics_qta(df: pd.DataFrame):
    # date_freq = df["BasketDate"].dt.date.value_counts() 
    # date_freq.plot()

    qta_freq = df["Qta"].value_counts()
    tmp_min = df["Qta"].min()
    tmp_max = df["Qta"].max()

    bins = list(range(tmp_min, tmp_max+1, 10))
    df['binned'] = pd.cut(df['Qta'], bins)
    qta_freq = df['binned'].value_counts()
    print(qta_freq)

    g = qta_freq.groupby( lambda x : x  )

    b = pd.DataFrame([group.to_numpy() for _, group in g]).transpose()
    print(g)
    quit()
    """
    labels = [f"{tmp[0]}/{tmp[1]}" for tmp in g.groups.keys()]
    b.columns = labels
    """
    b.boxplot(rot=-30)
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

    df = pd.read_csv('customer_supermarket_2.csv', index_col=0, parse_dates=["BasketDate"])

    df["Sale"].plot.box()
    plt.show()

    # df["Qta"].plot.box()
    # df["Qta"].corr(df["Sale"])

    # === Calculate statistical informations ===
    """
    statistics_basketID(df)
    statistics_basketDate(df)
    statistics_qta(df)
    """
    
    # print(list(sorted(list(df["CustomerCountry"].unique()))))

    # print(df['CustomerCountry'].unique())
    # print(df[df['CustomerCountry'].str.contains("Unspecified")].to_string())

    # print(df['Qta'].unique())
