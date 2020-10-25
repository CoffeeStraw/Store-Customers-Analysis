"""
Data Mining Project
Authors: CoffeeStraw, sd3ntato
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from math import log


def fix_dataset(df: pd.DataFrame):
    """
    Performs some data quality operations on the dataset
    """
    pd.set_option('mode.chained_assignment', None)

    # Converts sale to float, accomodating the csv format
    df["Sale"] = df["Sale"].str.replace(',', '.').astype(float)

    # Remove unidentified customers and converts CustomerID to int
    df.dropna(subset=['CustomerID'], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Put all characters in uppercase and remove extra whitespaces for products' description
    df["ProdDescr"] = df["ProdDescr"].str.upper().str.strip()

    # Put all characters in uppercase for product ids
    df["ProdID"] = df["ProdID"].str.upper()

    # Remove purchases with prices less or equal to zero, together with some outliers that costs less than 0.01
    df = df[df["Sale"] >= 0.01]

    # Remove C from basketIDs, since it is pointless (we already have negative quantities to identify those)
    # NOTE: We also previously dropped baskets starting with 'A', which had negative sale
    df["BasketID"] = df["BasketID"].str.replace('C', '').astype(int)

    # Uniform descriptions of same productIDs by taking the longest (more informations)
    tmp = df.groupby(["ProdID"]).nunique()["ProdDescr"].eq(1)
    tmp = tmp[tmp == False].index
    new_prod_descr = df[df["ProdID"].isin(tmp)].groupby("ProdID").aggregate({'ProdDescr': 'max'})

    def uniform_descr(x):
        if x.loc["ProdID"] in new_prod_descr.index:
            descr = new_prod_descr.loc[x.loc["ProdID"]]["ProdDescr"]
            x.loc["ProdDescr"] = descr
        return x

    df[["ProdID", "ProdDescr"]] = df[["ProdID", "ProdDescr"]].apply(uniform_descr, axis=1)

    # Put multiple products in the same basket as a single product (only ones with same sale)
    df = df.groupby(['BasketID','ProdID', 'Sale']).agg({
        'BasketDate': 'min',
        'Qta': 'sum',
        'CustomerID': 'min',
        'CustomerCountry': 'min',
        'ProdDescr': 'min'
    }).reset_index()
    
    # Remove products in the same basket, but having different sales
    def fun(x):
        if len(x) > 1:
            if x["Qta"].iloc[0] > 0 and x["ProdID"].iloc[0] != "M":
                return x

    tmp = df.groupby(['BasketID','ProdID']).apply(fun).drop_duplicates().dropna()[["BasketID", "ProdID"]]
    df = df.drop(tmp.index)

    # Tuple che hanno stesso ID oggetto e ID utente, ma hanno prima quantità positiva, poi negativa
    # (quindi quelle negative hanno indice maggiore)
    # Non credo la farò, mi sembrano comunque dati utili
    """
    to_delete = []
    def bleah(x):
        # Get products bought and sold by the same customer
        tmp = pd.merge( x[x["Qta"] < 0], x[x["Qta"] > 0], how='inner', on=['ProdID'] )["ProdID"]
        # Get entries containing the pre-calculated products and order by date
        tmp = x[ x["ProdID"].isin(tmp) ].sort_values("BasketDate")

        if len(tmp.index) == 0:
            return None

        if tmp["Qta"].iloc[0] < 0:
            print(tmp)
            print("=====================")

    tmp = df.groupby( ["CustomerID", "ProdID"] ).apply(bleah)
    print(tmp)
    """

    # Plot outliers
    plt.rcParams['figure.figsize'] = 10, 10
    df["Sale"].plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Sale")
    plt.clf()
    
    df["Qta"].plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Qta")
    plt.clf()

    df[abs(df["Sale"])<30]["Sale"].value_counts().sort_index().plot()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Sale_Distribution")
    plt.clf()

    df[abs(df["Qta"])<100]["Qta"].value_counts().sort_index().plot()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Qta_Distribution")
    plt.clf()

    # === REMOVE THE OUTLIERS ===
    def iqr_non_outliers(s: pd.Series):
        """
        Returns a true-list of the outliers in a column of the DataFrame,
        based on the quantiles
        """
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        trueList = ~((s < (Q1 - 1.5 * IQR)) |(s > (Q3 + 1.5 * IQR)))
        return trueList
    
    # Remvoe outliers from Qta with IQR method
    df = df[iqr_non_outliers(df["Qta"])]
    # Remove outliers from Sale based on Z-Score
    df = df[np.abs(stats.zscore(df["Sale"])) < 3]

    # Remove BasketIDs outliers based on zscore
    df[["BasketID", "Qta"]].groupby("BasketID").agg('sum').plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_BasketID")
    plt.clf()

    df[["BasketID", "Qta"]].groupby("BasketID").agg('sum')["Qta"].value_counts().sort_index().plot()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_BasketID_Distribution")
    plt.clf()

    tmp = df[["BasketID", "Qta"]].groupby("BasketID").agg('sum')
    tmp = tmp[np.abs(stats.zscore(tmp)) < 3].index
    df = df[df["BasketID"].isin(tmp)]

    # Rename columns with names that could mislead
    df.rename(columns={'CustomerCountry': 'PurchaseCountry', 'BasketDate': 'PurchaseDate'}, inplace=True)

    # Sort by date the dataset and reset indexes
    df.sort_values("PurchaseDate", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Swap columns
    df = df[["BasketID", "ProdID", "Sale", "Qta", "ProdDescr", "PurchaseDate", "PurchaseCountry", "CustomerID"]]

    # Save the pre-processed dataset
    df.to_csv("customer_supermarket_2.csv")
    pd.set_option('mode.chained_assignment','warn')


def customer_profilation(df: pd.DataFrame):
    """Create a new dataset with a profilation of each customer.
    """
    # TODO
    groups = df[df["Qta"]>0].groupby("CustomerID") 
    cdf = pd.DataFrame(data=np.array( [
        [
        group[0],
        sum(group[1]["Qta"]), # totale oggetti comprati
        sum(group[1]["Sale"]*group[1]["Qta"]), # totale soldi spesi
        len(group[1]["ProdID"].unique()), # numero oggetti distinti
        max( [ sum( g[1]["Qta"] ) for g in group[1].groupby("BasketID") ] ), # massimo numero oggetti acquistati in una shopping session
        max( [ sum( g[1]["Sale"]*g[1]["Qta"] ) for g in group[1].groupby("BasketID") ] ), # massima spesa carrello
        np.mean( [ sum( g[1]["Sale"]*g[1]["Qta"] ) for g in group[1].groupby("BasketID") ] ), # spesa media carrello
        np.mean( [ sum( g[1]["Qta"] ) for g in group[1].groupby("BasketID") ] ), # media oggetti in carrello
        group[1].groupby('ProdID').aggregate({'Qta':'sum'}).idxmax()[0] #oggetto preferito
        ] for group in groups 
        ] ), columns=["CustomerID","TotalItems","TotalSale","DistinctItems","MaxItems","MaxSale","MeanSale","MeanArticles","PreferedItem"] )
    cdf.to_csv("customer_profilation.csv")

def shannon_entropy( X : pd.Series ):
    e = 0
    m = len(X)
    for mi in X.value_counts():
        e += (mi/m)*log( (mi/m) ,2)
    return -e

def joint_entropy( X : pd.Series, Y : pd.Series ):
    e = 0
    df = pd.DataFrame( X ).join( Y );df.columns = ["uno","due"]
    groups = df.groupby(['uno','due'])
    m = len(groups.groups)
    for g in groups:
        mi = len(g[1])
        e += (mi/m)*log( (mi/m) ,2)
    return -e

def mutual_information( X : pd.Series, Y : pd.Series ):
    return shannon_entropy(X)+shannon_entropy(Y)-joint_entropy(X,Y)


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


def semantical_analysis(df: pd.DataFrame):
    """Perform some variable wise checks to understand the dataset.
    """
    # PRE: Check if basket starting with 'C' all have quantity less than 0
    # Results: only basket starting with 'C' have quantity less than 0
    # tmp = df[ (df["BasketID"].str.contains('C')) & (df["Qta"] > 0) ]
    # tmp = df[ df["Qta"] < 0 ]

    # Check if we have the same product inside the same basket
    # Result: two cases, same price, different price
    def check_duplicated_prods(x):
        if len(x) > 1:
            if x["Sale"].nunique() == 1:
                print(x)
    df.groupby(['BasketID','BasketDate','ProdID']).apply(check_duplicated_prods)

    # Check if with same BasketID we have different datetimes
    # Results: change BasketDate to PurchaseDate
    tmp = df.groupby(["BasketID"]).nunique()["BasketDate"].eq(1)
    tmp = tmp[tmp == False]

    # Check if two customers happend to have the same BasketID
    # Result: after removing duplicates no other wrong value found
    tmp = df.groupby(["BasketID", "CustomerID"]).ngroups

    tmp = df.groupby(["BasketID"]).ngroups

    tmp = df.groupby(["BasketID"]).nunique()["CustomerID"].eq(1)
    tmp = tmp[tmp == False].index

    # Check if discount are always alone in the basket
    # Result: Almost always, only one time we have it together with Manual
    tmp = df[
        df["BasketID"].isin(df[df['ProdID'] == "D"]["BasketID"])
    ]
    tmp = tmp[tmp["ProdID"] != "D"]["ProdDescr"]
    
    # Check if baskets only are numerical with an optional starting 'C' character
    # Result: We found baskets starting with 'A', which however will be removed since they have sales less than 0
    tmp = df[~df['BasketID'].str.contains('C')][df['BasketID'].str.contains('[A-Za-z]')]["BasketID"].unique()

    # Check for strange ProductID (nor alphanumerical code only)
    # Result: A lot of products contains characters, we get to know about discounts and bank charges
    tmp = df[df['ProdID'].str.contains('[A-Za-z]')]["ProdID"].unique()

    # Check for non-uppercase descriptions
    # Result: we get to know about descriptions being inconsistent
    tmp = df[df['ProdDescr'].str.contains('[a-z]')]["ProdDescr"].unique()

    # Check list of countries
    # Result: (Get to know about hidden null-values: 'Unknown')
    tmp = list(sorted(list(df["CustomerCountry"].unique())))

    # Check for strange qta values
    # Result: Get to know about negative values and outliers
    tmp = df['Qta'].unique()

    # CustomerCountry seems like the country where the user registered... is that true?
    # Result: no, since some IDs have different countries. Change name to 'PurchaseCountry'
    tmp = df.groupby(["CustomerID"]).nunique()["CustomerCountry"].eq(1)

    # Do all ProdID have one ProdDescr?
    # Result: No, some descriptions are more verbose, we will take those
    tmp = df.groupby(["ProdID"]).nunique()["ProdDescr"].eq(1)
    tmp = tmp[tmp == False].index
    tmp = df[df["ProdID"].isin(tmp)].groupby("ProdID").aggregate({'ProdDescr': 'max'})

    # Do we have sales with more than 3 digit places?
    tmp = df["Sale"].astype(str).str.contains(r"\.\d{3}").value_counts()


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

    df = pd.read_csv('customer_supermarket_2.csv', index_col=0, parse_dates=["PurchaseDate"])
    print(df)
    quit()

    """
    s = pd.Series( [g[1]["BasketID"].nunique() for g in df.groupby("PurchaseCountry")], index=[g[0] for g in df.groupby("PurchaseCountry")] )
    print(s)

    print(sum(s))
    print(s['United Kingdom'], s['United Kingdom'] / s.sum() * 100)
    quit()
    s = s.drop('United Kingdom')
    #s.sort_values()
    #s.drop(s[ s<30 ].index).plot(kind='pie',rotatelabels=True)
    s['others'] = s[ s<30 ].sum()

    s = s.drop(s[ s<30 ].index) 
    s.plot(kind='pie',rotatelabels=True)
    """

    # === Perform a semantical analysis of the dataset ===
    # semantical_analysis(df)

    # df["Sale"].plot.box()
    # plt.show()

    # df["Qta"].plot.box()
    # df["Qta"].corr(df["Sale"])

    # === Calculate statistical informations ===
    """
    statistics_basketID(df)
    statistics_basketDate(df)
    statistics_qta(df)
    """

    """ ROBA VALE
    pd.DataFrame(data=np.array( [
             [
              group[0],
              sum(group[1]["Qta"]), # somma tutti oggetti
              len(group[1]["ProdID"].unique())  # numero oggetti distinti
              max( [ sum( g[1]["Qta"] ) for g in group[1].groupby("BasketID") ] ) # massimo numero oggetti acquistati in una shopping session
             ] for group in groups 
             ] ), columns=["CusomerID","TotalItems","DistinctItems","MaxItems"]  )

    from math import log
    # total number of observations m : m
    # number of observation of category i : mi

    def shannon_entropy(X : pd.Series):
        e = 0
        m = len(X)
        for mi in X.value_counts():
        e += (mi/m)*log( (mi/m) ,2)
        return -e
    """