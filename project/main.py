"""
Data Mining Project
Authors: CoffeeStraw, sd3ntato
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from math import log, ceil
from natsort import natsorted
import seaborn as sn


def plt_radar(df: pd.DataFrame, filepath=""):
    """Represent a DataFrame using a radar plot.
    """
    # Number of variable
    categories=list(df.index)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ylim = ceil(df.max().max())
    ticks = list(range(0,ylim,5))
    ticks_str = list(map(lambda x: str(x), ticks))
    plt.yticks(ticks, ticks_str, color="grey", size=7)
    plt.ylim(0,ylim)

    # PART 2: Add plots
    # Ind1
    values = list(df[df.columns[0]])
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[0])
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    values = list(df[df.columns[1]])
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[1])
    ax.fill(angles, values, 'r', alpha=0.1)
    
    # Add legend and tight the layout
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()

    # Show or save?
    if not filepath:
        plt.show()
        plt.clf()
    else:
        plt.savefig(filepath)


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


def fix_dataset(df: pd.DataFrame):
    """
    Performs some data quality operations on the dataset
    """
    # Converts sale to float, accomodating the csv format
    df["Sale"] = df["Sale"].str.replace(',', '.').astype(float)

    # Remove unidentified customers and converts CustomerID to int
    df.dropna(subset=['CustomerID'], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Put all characters in uppercase and remove extra whitespaces for products' description
    df["ProdDescr"] = df["ProdDescr"].str.upper().str.strip()

    # Put all characters in uppercase for product ids
    df["ProdID"] = df["ProdID"].str.upper()

    # Remove purchases with prices less than or equal to zero, together with some outliers that costs less than 0.01
    df = df[df["Sale"] >= 0.01]

    # Remove C from basketIDs, since it is pointless (we already have negative quantities to identify those)
    # NOTE: We also previously dropped baskets starting with 'A', which had negative sale
    df["BasketID"] = df["BasketID"].str.replace('C', '').astype(int)

    # Uniform descriptions of same productIDs by taking the longest (more informations)
    tmp = df.groupby(["ProdID"]).nunique()["ProdDescr"].eq(1)
    tmp = tmp[tmp == False].index
    new_prod_descr = df[df["ProdID"].isin(tmp)].groupby("ProdID").agg({'ProdDescr': 'max'})

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

    # === REMOVE THE OUTLIERS ===
    plt.rcParams['figure.figsize'] = 10, 10
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

    # Remove outliers from Sale based on Z-Score
    df_sale = df["Sale"]

    df_sale.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Sale")
    plt.clf()

    df_sale[df_sale < 50].hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Sale_Distribution")
    plt.clf()

    df = df[abs(stats.zscore(df_sale)) < 3]

    # Remvoe outliers from Qta with IQR method
    df_qta = df["Qta"]

    df_qta.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Qta")
    plt.clf()

    df_qta[abs(df_qta) < 100].hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Qta_Distribution")
    plt.clf()

    df = df[iqr_non_outliers(df_qta)]

    # Remove outliers from BasketID based on zscore
    df_articles_per_basket = df[["BasketID", "Qta"]].groupby("BasketID").agg('sum')["Qta"]

    df_articles_per_basket.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_BasketID")
    plt.clf()

    df_articles_per_basket[abs(df_articles_per_basket) < 2000].hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_BasketID_Distribution")
    plt.clf()

    bid_good = df_articles_per_basket[abs(stats.zscore(df_articles_per_basket)) < 3].index
    df = df[df["BasketID"].isin(bid_good)]

    # Remove outliers from CustomerID based on zscore
    df_totsale_per_user = pd.Series([round( sum( g[1]["Sale"]*g[1]["Qta"] ), 2) for g in df.groupby('CustomerID')], index=[g[0] for g in df.groupby('CustomerID')])
    
    df_totsale_per_user.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_CustomerID")
    plt.clf()

    df_totsale_per_user.hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_CustomerID_Distribution")
    plt.clf()

    cid_good = df_totsale_per_user[abs(stats.zscore(df_totsale_per_user)) < 3].index
    df = df[df["CustomerID"].isin(cid_good)]

    # Drop rows corresponding to returns without relative purchase (inconsistent data)
    invalid_indexes = []
    def get_invalid_indexes(x):
        x = x.sort_values(by='BasketDate')
        s = 0
        for i, qta in enumerate(x['Qta']):
            if (s := s + qta) < 0:
                invalid_indexes.append(x.iloc[i].name)

    df[ ~df["ProdID"].isin(['M', 'D', 'BANK CHARGES']) ].groupby(['CustomerID', 'ProdID']).apply(get_invalid_indexes)
    print("N. of dropped inconsistent returns:", len(invalid_indexes))
    df.drop(invalid_indexes, inplace=True)

    # Rename columns with names that could mislead
    df.rename(columns={'CustomerCountry': 'PurchaseCountry', 'BasketDate': 'PurchaseDate'}, inplace=True)

    # Swap columns
    df = df[["BasketID", "ProdID", "Sale", "Qta", "ProdDescr", "PurchaseDate", "PurchaseCountry", "CustomerID"]]

    # Sort by date the dataset and reset indexes
    df.sort_values("PurchaseDate", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save the pre-processed dataset
    df.to_csv("customer_supermarket_2.csv")


def distribution_and_statistics(df: pd.DataFrame):
    # Sale statistics
    print("SALE DESCRIBE:", df["Sale"].describe())

    # Sale distribution
    df_products_catalog = df[["ProdID", "Sale"]].drop_duplicates()["Sale"]
    print("PRODUCTS CATALOG DESCRIBE:", df_products_catalog.describe())

    df_products_catalog.hist(bins=50)
    plt.tight_layout()
    plt.savefig("../report/imgs/Sale_Distribution")
    plt.clf()

    df_products_catalog.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Sale_Box_Plot")
    plt.clf()

    # Distribution of buys and returns
    print("RATION QTA POSITIVE/NEGATIVE:", (df["Qta"] > 0).value_counts())
    print("STATISTICS QTA > 0:", df[df["Qta"] > 0]["Qta"].describe())
    print("STATISTICS QTA < 0:", df[df["Qta"] < 0]["Qta"].describe())

    df.plot.scatter('Qta', 'Sale', c='Sale', colormap='winter', colorbar=False, figsize=(10,7))
    plt.tight_layout()
    plt.savefig("../report/imgs/Sale_Qta_Distribution")
    plt.clf()

    # === Monthly statistics ===
    def year_month(i):
        x = df.loc[i]['PurchaseDate']
        return f"{x.year}/{x.month}"

    # Number of baskets and profit per month
    tmp = df[["PurchaseDate", "Sale", "Qta"]]
    tmp["Profit"] = tmp["Sale"] * tmp["Qta"]
    tmp.drop(["Sale", "Qta"], axis=1, inplace=True)
    monthly_stats = tmp.groupby(year_month).agg('sum')

    monthly_stats["Baskets"] = df[["PurchaseDate", "BasketID"]].drop_duplicates().groupby(year_month).size()
    monthly_stats = monthly_stats.reindex(index=natsorted(monthly_stats.index))

    print("MONTHLY STATS:", monthly_stats)
    print("CORRELAZIONE:")
    print(monthly_stats.corr())

    monthly_stats['Baskets'] = monthly_stats['Baskets'] / sum(monthly_stats['Baskets']) * 100
    monthly_stats['Profit'] = monthly_stats['Profit'] / sum(monthly_stats['Profit']) * 100

    monthly_stats.plot.bar(figsize=(14,7))
    plt.tight_layout()
    plt.legend(loc=2, prop={'size': 15})
    plt.savefig("../report/imgs/Monthly_Baskets_Profit")
    plt.clf()

    # Number of baskets and profit per country
    tmp = df[["PurchaseCountry", "Sale", "Qta"]]
    tmp["Profit"] = tmp["Sale"] * tmp["Qta"]
    tmp.drop(["Qta", "Sale"], axis=1, inplace=True)
    country_stats = tmp.groupby("PurchaseCountry").agg('sum')

    country_stats["Baskets"] = df[['PurchaseCountry', 'BasketID']].groupby('PurchaseCountry').agg(lambda x: x.nunique())['BasketID']

    print("COUNTRY STATS:", country_stats)
    print("CORRELAZIONE:")
    print(country_stats.corr())

    # We prepare two plots: the first one using only UK while grouping the other, the second one is without UK
    tmp = country_stats[country_stats.index != 'United Kingdom'].agg('sum')
    country_stats1 = country_stats[country_stats.index == 'United Kingdom']
    country_stats1.loc["Others"] = tmp.values
    
    # Normalize values
    country_stats1['Baskets'] = country_stats1['Baskets'] / sum(country_stats1['Baskets']) * 100
    country_stats1['Profit'] = country_stats1['Profit'] / sum(country_stats1['Profit']) * 100

    country_stats1.plot.bar(figsize=(4,7))
    plt.tight_layout()
    plt.savefig("../report/imgs/Country_Baskets_Profit")
    plt.clf()

    # Remove United Kingdom
    country_stats2 = country_stats.drop('United Kingdom')
    # Aggregate small values
    threshold = 25
    tmp = country_stats2[country_stats2["Baskets"] < threshold].agg('sum')
    country_stats2 = country_stats2[country_stats2["Baskets"] >= threshold]
    country_stats2.loc["Others"] = tmp.values
    # Normalize values
    country_stats2['Baskets'] = country_stats2['Baskets'] / sum(country_stats2['Baskets']) * 100
    country_stats2['Profit'] = country_stats2['Profit'] / sum(country_stats2['Profit']) * 100

    plt_radar(country_stats2, "../report/imgs/Country_Basket_Profit_No_UK")

    # Countries per month (scrivi meglio)
    # df.groupby([year_month, 'PurchaseCountry']).agg(lambda x: print(x))
    # quit()

    # Most popular products
    popular_prods = df[['ProdDescr', 'Qta']].groupby('ProdDescr').agg('sum').sort_values(by='Qta', ascending=False).head(30)
    popular_prods.plot.barh(figsize=(20,10), color='darkred')
    plt.tight_layout()
    plt.savefig("../report/imgs/Products_Popular")
    plt.clf()


def customer_profilation(df: pd.DataFrame):
    """Create a new dataset with a profilation of each customer.
    """
    def entropy(g):
        l = g[["ProdID", "Qta"]].groupby('ProdID').agg('sum')
        m = l.values.sum()
        e = -sum( [ (mi/m)*log((mi/m), 2) for mi in l.values.flatten() ] )
        return round(e, 2)
    
    # Total purchased items
    l = lambda g: sum( g["Qta"] )
    # Number of distinct items
    lu = lambda g: len( g["ProdID"].unique() )
    # Maximum number of purchased items in a shopping session
    lmax = lambda g: max( [ sum( g1[1]["Qta"] ) for g1 in g.groupby("BasketID") ] )
    # Total money spent
    tot_sale = lambda g: round( sum( g["Sale"]*g["Qta"] ), 2)
    # Max amount for a basket
    max_sale = lambda g: round( max( [ sum( g1[1]["Sale"]*g1[1]["Qta"] ) for g1 in g.groupby("BasketID") ] ), 2)
    # Medium amount for a basket
    mean_sale = lambda g: round( np.mean( [ sum( g1[1]["Sale"]*g1[1]["Qta"] ) for g1 in g.groupby( "BasketID" ) ] ), 2)
    # Medium object in basket
    mean_items = lambda g: int( np.mean( [ sum( g1[1]["Qta"] ) for g1 in g.groupby("BasketID") ] ))
    # Preferred item
    preferred_item = lambda g: g.groupby('ProdID').agg({'Qta':'sum'}).idxmax()[0]
    # Main country
    main_country = lambda g: g[['BasketID','PurchaseCountry']].groupby('PurchaseCountry').nunique().idxmax()[0] 
    
    groups = df[df["Qta"]>0].groupby("CustomerID")
    cdf = pd.DataFrame(data=np.array( [
        [
        group[0],
        l(group[1]),
        lu(group[1]),
        lmax(group[1]), 
        entropy(group[1]),
        tot_sale(group[1]),
        max_sale(group[1]),
        mean_sale(group[1]),
        mean_items(group[1]),
        preferred_item(group[1]),
        main_country(group[1])
        ] for group in groups
    ] ), columns=["CustomerID","l","lu","lmax","E","TotSale","MaxSale","MeanSale","MeanItems","PrefItem","MainCountry"] )
    cdf.set_index('CustomerID',inplace=True)

    # Workaround for Pandas' bug (not able to convert to correct dtypes)
    # cdf.convert_dtypes()
    cdf.to_csv("customer_profilation.csv")
    cdf = pd.read_csv("customer_profilation.csv", index_col=0)

    # calculate percentage of returned item for customer
    groups = df[ (df["Qta"]<0) & ~(df["ProdID"].isin(['M', 'D', 'BANK CHARGES'])) ][['CustomerID','Qta']].groupby("CustomerID").agg('sum')
    cdf['PReturn'] = pd.Series(
        [ round(-groups.loc[i]['Qta']/cdf.loc[i]['l']*100, 2) if i in groups.index else 0 for i in cdf.index ],
        index=cdf.index
    )
    print(cdf)
    cdf.to_csv("customer_profilation.csv")


if __name__ == "__main__":
    pd.set_option('mode.chained_assignment', None)

    """
    df = pd.read_csv('customer_supermarket.csv', sep='\t', index_col=0, parse_dates=["BasketDate"])

    # Prints data's samples and informations,
    # including the number of not null values for each columns
    print(df.info(), "\n")
    print(df.head())

    fix_dataset(df)
    quit()
    """

    df = pd.read_csv('customer_supermarket_2.csv', index_col=0, parse_dates=["PurchaseDate"])

    # === Data Distribution & Statistics ===
    # distribution_and_statistics(df)

    # === CUSTOMER PROFILATION ===
    # customer_profilation(df)
    cdf = pd.read_csv('customer_profilation.csv')

    """
    sn.heatmap(cdf.drop('CustomerID', axis=1).corr(), annot=True)
    plt.show()
    plt.clf()

    pd.plotting.scatter_matrix(cdf,figsize=(15,15))
    plt.show()
    plt.clf()
    """