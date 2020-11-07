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


def plot(filename=""):
    """Conveniency function to show or save a plot
    """
    plt.tight_layout()
    if filename:
        plt.savefig(f"../report/imgs/{filename}")
    else:
        plt.show()
    plt.close()


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
        plt.close()
    else:
        plt.savefig(filepath)


def semantical_analysis(df: pd.DataFrame):
    """Perform some variable wise checks to understand the dataset.
    """
    # PRE: Check if basket starting with 'C' all have quantity less than 0
    # Results: only basket starting with 'C' have quantity less than 0
    tmp = df[ (df["BasketID"].str.contains('C')) & (df["Qta"] > 0) ]
    print("N. BasketID STARTING WITH 'C' AND Qta > 0:", len(tmp))
    
    tmp = df[ (df["Qta"] < 0) & ~(df["BasketID"].str.contains('C')) ]
    tmp.dropna(subset=['CustomerID'], inplace=True)
    print("N. BasketID WITH Qta < 0:", len(tmp))

    # Check if we have the same product inside the same basket
    # Result: two cases, same price, different price
    done = False
    def check_duplicated_prods(x):
        nonlocal done
        if len(x) > 1 and x["Sale"].nunique() == 1 and not done:
            print("BASKET WITH INCONSISTENT Qta:\n", x)
            done = True
    df.groupby(['BasketID','BasketDate','ProdID']).apply(check_duplicated_prods)

    # Check if with same BasketID we have different datetimes
    # Results: change BasketDate to PurchaseDate
    tmp = df.groupby(["BasketID"]).nunique()["BasketDate"].eq(1)
    tmp = tmp[tmp == False]
    print("INCONSISTENT BasketDates:", len(tmp))

    # Check if two customers happen to have the same BasketID
    # Result: after removing duplicates no other wrong value found
    tmp = df.groupby(["BasketID", "CustomerID"]).ngroups
    print("N. BasketID-CustomerID COUPLES:", tmp)

    tmp = df["BasketID"].nunique()
    print("N. BasketID:", tmp)

    tmp = df.dropna(subset=['CustomerID'])
    tmp = tmp.groupby(["BasketID"]).nunique()["CustomerID"].eq(1)
    tmp = tmp[tmp == False].index
    print("INCONSITENT BasketID-CustomerID (after NaN removal):", len(tmp))

    # Check if discount are always alone in the basket
    # Result: Almost always, only one time we have it together with Manual
    tmp = df[
        df["BasketID"].isin(
            df[df['ProdID'] == "D"]["BasketID"]
    )]
    tmp = tmp[tmp["ProdID"] != "D"]
    print("PRODUCTS IN THE SAME BASKET WITH DISCOUNT:\n", tmp)
    
    # Check if baskets only are numerical with an optional starting 'C' character
    # Result: We found baskets starting with 'A', which however will be removed since they have sales less than 0
    tmp = df[~df['BasketID'].str.contains('C')][df['BasketID'].str.contains('[A-Za-z]')]["BasketID"].unique()
    print("STRANGE BASKETS:\n", tmp)

    # Check for strange ProductID (nor alphanumerical code only)
    # Result: A lot of products contains characters, we get to know about discounts and bank charges
    tmp = df[df['ProdID'].str.contains('[A-Za-z]')]["ProdID"].unique()
    print("STRANGE ProductID:\n", tmp)

    # Check for non-uppercase descriptions
    # Result: we get to know about descriptions being inconsistent and some strange descriptions
    tmp = df['ProdDescr'].isna().sum()
    print("N. NaN ProdDescr:", tmp)

    tmp = df.dropna(subset=['ProdDescr'])
    tmp = tmp[tmp['ProdDescr'].str.contains('[a-z]')]["ProdDescr"].unique()
    print("INCONSISTENT ProdDescr:\n", tmp)

    # Check list of countries
    # Result: (Get to know about hidden null-values: 'Unspecified')
    tmp = list(sorted(list(df["CustomerCountry"].unique())))
    print("COUNTRIES:", tmp)

    # Check for strange qta values
    # Result: Get to know about negative values and outliers
    tmp = df['Qta'].describe()
    print("Qta Descr:", tmp)

    # CustomerCountry seems like the country where the user registered... is that true?
    # Result: no, since some IDs have different countries. Some customers may have changed their nationality.
    # We will take this into account when we will create the customer profilation dataset.
    tmp = df.groupby(["CustomerID"]).nunique()["CustomerCountry"].eq(1)
    tmp = list(tmp[tmp == False].index)
    print("INCONSISTENT CustomerCountry:", tmp)

    # Do all ProdID have one ProdDescr?
    # Result: No, some descriptions are more verbose, we will take those
    tmp = df.groupby(["ProdID"]).nunique()["ProdDescr"].eq(1)
    tmp = tmp[tmp == False].index
    print("N. INCONSISTENT ProdDescr:", len(tmp))

    # Do we have sales with more than 3 digit places?
    # Result: Yes, we will round them
    tmp = df["Sale"].astype(str).str.contains(r",\d{3,}")
    tmp = tmp[tmp == True].index
    tmp = df.loc[tmp]
    print("INCONSISTENT Sale:\n", tmp)


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
    # We remove them since they're few (4)
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

    # Put multiple products in the same basket as a single product
    df = df.groupby(['BasketID','ProdID']).agg({
        'BasketDate': 'min',
        'Qta': 'sum',
        'Sale': 'mean',
        'CustomerID': 'min',
        'CustomerCountry': 'min',
        'ProdDescr': 'min'
    }).reset_index()

    # === REMOVE THE OUTLIERS ===
    plt.rcParams['figure.figsize'] = 10, 10
    def iqr_non_outliers(s: pd.Series):
        """Returns a true-list of the outliers in a column
        of the DataFrame, based on the quantiles"""
        
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)

        IQR = Q3 - Q1
        trueList = (s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))
        return trueList

    # Outliers in ARTICLES from SALE
    df_sale = df['Sale']

    df_sale.plot.box()
    plot("../report/imgs/Outliers_Sale")
    df_sale[df_sale < 50].hist(bins=100)
    plot("../report/imgs/Outliers_Sale_Distribution")

    # Search for a threshold and remove based on that
    df[df_sale <= 5000].plot.box()
    plot()
    df[df_sale <= 2000].plot.box()
    plot()
    df = df[df_sale <= 1000]
    quit()

    # Outliers in ARTICLES from QTA
    df_qta = df["Qta"]

    df_qta.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Qta")
    plt.close()

    df_qta[abs(df_qta) < 100].hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_Qta_Distribution")
    plt.close()

    # Would IQR be effective?
    # Result: no, since we think that most of the customers are wholesalers and it would drop too many entries
    qta_iqr_outliers = df_qta[~iqr_non_outliers(df_qta)]
    print("QTA - IQR RESULTS:\n", qta_iqr_outliers.describe())
    print("MIN Qta Positives:", qta_iqr_outliers[qta_iqr_outliers > 0].min())
    print("MAX Qta Negatives:", qta_iqr_outliers[qta_iqr_outliers < 0].max())
    
    # Solution: remove based on a threshold. But first one last check: how are those outliers distributed among the users?
    # TODO: fai vedere i boxplot
    outliers = df.loc[df_qta[abs(df_qta) >= 3500].index]
    print("QTA OUTLIERS (with threshold of 3500):")
    print(outliers["Qta"].describe())
    print(outliers["CustomerID"].nunique())

    # Values come from different users, we cannot just drop the customers, must instead drop single tuples
    df["Qta"] = df_qta[abs(df_qta) < 3500]

    # Remove outliers from BasketID based on IQR
    df_articles_per_basket = df[["BasketID", "Qta"]].groupby("BasketID").agg('sum')["Qta"]

    df_articles_per_basket.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_BasketID")
    plt.close()

    df_articles_per_basket[abs(df_articles_per_basket) < 2000].hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_BasketID_Distribution")
    plt.close()

    # Check if IQR 
    basketid_iqr_outliers = df_articles_per_basket[~iqr_non_outliers(df_articles_per_basket)]
    print("BASKETID - IQR RESULTS:\n", basketid_iqr_outliers.describe())
    print("MIN BASKETID-QTA Postives:", basketid_iqr_outliers[basketid_iqr_outliers > 0].min())
    print("MAX BASKETID-QTA Negatives:", basketid_iqr_outliers[basketid_iqr_outliers < 0].max())

    # Remove outliers based on a threshold
    # TODO: Fai vedere i boxplot
    non_outliers = df_articles_per_basket[abs(df_articles_per_basket) < 5000]
    df = df[df["BasketID"].isin(non_outliers.index)]

    df_articles_per_basket = df[["BasketID", "Qta"]].groupby("BasketID").agg('sum')["Qta"]

    # Remove outliers from CustomerID

    # Remove outliers from CustomerID based on IQR
    df_totsale_per_user = pd.Series([round( sum( g[1]["Sale"]*g[1]["Qta"] ), 2) for g in df.groupby('CustomerID')], index=[g[0] for g in df.groupby('CustomerID')])
    
    df_totsale_per_user.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_CustomerID")
    plt.close()

    df_totsale_per_user.hist(bins=100)
    plt.tight_layout()
    plt.savefig("../report/imgs/Outliers_CustomerID_Distribution")
    plt.close()

    # Try with IQR
    customerid_iqr_outliers = df_totsale_per_user[~iqr_non_outliers(df_totsale_per_user)]
    print("CUSTOMERID - IQR RESULTS:\n", customerid_iqr_outliers.describe())

    # Remove based on a threshold
    # TODO: Fai box plots (forse basta quello generato)
    non_outliers = df_totsale_per_user[abs(df_totsale_per_user) < 50000]
    df = df[df["CustomerID"].isin(non_outliers.index)]

    # Drop rows corresponding to returns without relative purchase (inconsistent data)
    invalid_indexes = []
    def get_invalid_indexes(x):
        x = x.sort_values(by='BasketDate')
        s = 0
        for i, qta in enumerate(x['Qta']):
            if (s := s + qta) < 0:
                invalid_indexes.append(x.iloc[i].name)
                s = 0
    
    df[ ~df["ProdID"].isin(['M', 'D', 'BANK CHARGES']) ].groupby(['CustomerID', 'ProdID']).apply(get_invalid_indexes)
    df.drop(invalid_indexes, inplace=True)
    print("N. of dropped inconsistent returns:", len(invalid_indexes))

    # Rename columns with names that could mislead
    df.rename(columns={'BasketDate': 'PurchaseDate'}, inplace=True)

    # Swap columns
    df = df[["BasketID", "ProdID", "ProdDescr", "Sale", "Qta", "PurchaseDate", "CustomerID","CustomerCountry"]]

    # Sort by date the dataset and reset indexes
    df.sort_values("PurchaseDate", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save the pre-processed dataset
    df.to_csv("customer_supermarket_2.csv")


def distribution_and_statistics(df: pd.DataFrame):
    # Sale statistics
    print("SALE DESCRIBE:\n", df["Sale"].describe())

    # Sale distribution
    df_products_catalog = df[["ProdID", "Sale"]].drop_duplicates()["Sale"]
    print("PRODUCTS CATALOG DESCRIBE:\n", df_products_catalog.describe())

    df_products_catalog.hist(bins=50)
    plt.tight_layout()
    plt.savefig("../report/imgs/Sale_Distribution")
    plt.close()

    df_products_catalog.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/Sale_Box_Plot")
    plt.close()

    # Distribution of buys and returns
    print("RATION QTA POSITIVE/NEGATIVE:\n", (df["Qta"] > 0).value_counts())
    print("STATISTICS QTA > 0:\n", df[df["Qta"] > 0]["Qta"].describe())
    print("STATISTICS QTA < 0:\n", df[df["Qta"] < 0]["Qta"].describe())

    df.plot.scatter('Qta', 'Sale', c='Sale', colormap='winter', colorbar=False, figsize=(10,7))
    plt.tight_layout()
    plt.savefig("../report/imgs/Sale_Qta_Distribution")
    plt.close()

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

    print("MONTHLY STATS:\n", monthly_stats)
    print("CORRELAZIONE:")
    print(monthly_stats.corr())

    monthly_stats['Baskets'] = monthly_stats['Baskets'] / sum(monthly_stats['Baskets']) * 100
    monthly_stats['Profit'] = monthly_stats['Profit'] / sum(monthly_stats['Profit']) * 100

    monthly_stats.plot.bar(figsize=(14,7))
    plt.tight_layout()
    plt.legend(loc=2, prop={'size': 15})
    plt.savefig("../report/imgs/Monthly_Baskets_Profit")
    plt.close()

    # Number of baskets and profit per country
    tmp = df[["CustomerCountry", "Sale", "Qta"]]
    tmp["Profit"] = tmp["Sale"] * tmp["Qta"]
    tmp.drop(["Qta", "Sale"], axis=1, inplace=True)
    country_stats = tmp.groupby("CustomerCountry").agg('sum')

    country_stats["Baskets"] = df[['CustomerCountry', 'BasketID']].groupby('CustomerCountry').agg(lambda x: x.nunique())['BasketID']

    print("COUNTRY STATS:\n", country_stats)
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
    plt.close()

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

    # Monthly activity per country
    ma_country = df.groupby(['CustomerCountry', year_month]).apply(lambda x: sum(x["Qta"] * x["Sale"]))
    ma_country = ma_country.unstack(level=0)
    
    ma_country = ma_country.reindex(index=natsorted(ma_country.index))
    cols = list(ma_country.columns)
    cols.sort(key=lambda x: ma_country[x].notnull().sum())
    ma_country = ma_country[cols]
    for i, c in enumerate(ma_country.columns):
        ma_country[c][ma_country[c].notnull()] = i

    f = ma_country.plot.line(figsize=(16,8), legend=False, style='-o')
    f.set_xticks(range(0, len(ma_country.index)))
    f.set_xticklabels([x.replace('/', '\n') for x in ma_country.index])
    f.set_yticks(range(0, len(ma_country.columns)))
    f.set_yticklabels(list(ma_country.columns))
    plt.tight_layout()
    plt.savefig("../report/imgs/Monthly_Activity_Country")
    plt.close()

    # Most popular products
    print("UNIQUE PRODUCTS:", len(df['ProdID'].unique()))

    popular_prods = df[['ProdDescr', 'Qta']].groupby('ProdDescr').agg('sum').sort_values(by='Qta', ascending=False).head(10)
    popular_prods.plot.barh(figsize=(10,3), color='darkred')
    plt.tight_layout()
    plt.savefig("../report/imgs/Products_Popular")
    plt.close()


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
    lu = lambda g: g["ProdID"].nunique()
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
    main_country = lambda g: g[['BasketID','CustomerCountry']].groupby('CustomerCountry').nunique().idxmax()[0] 
    # Number of baskets
    n_baskets = lambda g:  g['BasketID'].nunique()

    groups = df[df["Qta"]>0].groupby("CustomerID")
    cdf = pd.DataFrame(data=np.array( [
        [
        group[0],
        l(group[1]),
        lu(group[1]),
        lmax(group[1]), 
        entropy(group[1]),
        n_baskets(group[1]),
        tot_sale(group[1]),
        max_sale(group[1]),
        mean_sale(group[1]),
        mean_items(group[1]),
        preferred_item(group[1]),
        main_country(group[1])
        ] for group in groups
    ] ), columns=["CustomerID","l","lu","lmax","E","NBaskets","TotSale","MaxSale","MeanSale","MeanItems","PrefItem","MainCountry"] )
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
    cdf['SaleRate'] = cdf['TotSale']/cdf['l'] 

    print(cdf)
    cdf.to_csv("customer_profilation.csv")


def customer_statistics(cdf: pd.DataFrame):
    """
    Statistics obtained from customer profilation.
    """
    
    pd.plotting.scatter_matrix(cdf)

    print('---------- BASIC INFORMATION ----------')
    print( cdf.info() )
    print('---------- INDIVIDUAL ATTRIBUTE STATISTICS ----------')
    print( cdf.describe() )

    # Distrituion of numerical attributes with histograms
    cdf.hist(bins=50)
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_Histograms")
    plt.close()

    # Distrituion of numerical attributes with box-plots
    cdf.plot.box()
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_Box_Plots")
    plt.close()

    # Pairwise xorrelations with heatmap on correlation matrix
    _, ax = plt.subplots()
    sn.heatmap(cdf.corr(), cmap='coolwarm', annot=True, ax=ax)
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_HeatMap_Pairwise_Correlations")
    plt.close()

    # lu vs E
    cdf['log(lu)']=np.log(cdf['lu'])
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    cdf.plot.scatter('lu','E',c='l',colormap='viridis',ax=axes[0])
    cdf.plot.scatter('log(lu)','E',c='l',colormap='viridis' ,ax = axes[1]  )
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_lu_vs_E")
    plt.close()

    # l vs TotSale
    cdf.plot.scatter('l','TotSale',c='E',colormap='viridis')
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_l_vs_TotSale")
    plt.close()

    # l vs PReturn ( unico che droppa info nascosta: chi ha comprato tanti articoli solitamente non li riporta )
    cdf['1/PReturn']=np.reciprocal(cdf['PReturn'])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    cdf.plot.scatter('l','PReturn',c='E',colormap='viridis' ,ax = axes[0]  )
    cdf.plot.scatter('l','1/PReturn',c='E',colormap='viridis' ,ax = axes[1]  )
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_lu_vs_PReturn")
    plt.close()

    # TotSale vs MeanSale
    cdf.plot.scatter('TotSale','MeanSale',c='E',colormap='viridis')
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_TotSale_vs_MeanSale")
    plt.close()

    # l vs MeanItems
    cdf.plot.scatter('l','MeanItems',c='E',colormap='viridis')
    plt.tight_layout()
    plt.savefig("../report/imgs/cdf_l_vs_MeanItems")
    plt.close()


if __name__ == "__main__":
    pd.set_option('mode.chained_assignment', None)

    df = pd.read_csv('customer_supermarket.csv', sep='\t', index_col=0, parse_dates=["BasketDate"])

    # Prints data's samples and informations,
    # including the number of not null values for each columns
    # print(df.info(), "\n")
    # print(df.head())
    # semantical_analysis(df)

    fix_dataset(df)
    quit()
    """
    """

    # df = pd.read_csv('customer_supermarket_2.csv', index_col=0, parse_dates=["PurchaseDate"])

    # === Data Distribution & Statistics ===
    # distribution_and_statistics(df)
    
    # === CUSTOMER PROFILATION ===
    # customer_profilation(df)
    cdf = pd.read_csv('customer_profilation.csv', index_col=0)
    customer_statistics(cdf)
    quit()

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn import metrics
    from itertools import combinations

    # Reimpostazione dataframe
    # cdf = cdf[cdf['MainCountry'] == 'United Kingdom']
    attr_cluster = ['E','SaleRate','NBaskets']
    cdf = cdf[attr_cluster]
    # cdf['MainCountry'] = pd.factorize(cdf['MainCountry'])[0]
    # cdf['PrefItem'] = pd.factorize(cdf['PrefItem'])[0]

    def iqr_non_outliers(s: pd.Series):
        # Returns a true-list of the outliers in a column of the DataFrame,
        # based on the quantiles
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        trueList = ~((s < (Q1 - 1.5 * IQR)) |(s > (Q3 + 1.5 * IQR)))
        return trueList

    cdf = cdf[iqr_non_outliers(cdf['SaleRate'])]

    # normalizzazione (prova a denormalizzare)
    scaler = MinMaxScaler() # Minmax?
    X = scaler.fit_transform(cdf.values)

    #selezionare miglior valore di k
    """
    sse_list = list()
    max_k = 30
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
        kmeans.fit(X)
        
        sse = kmeans.inertia_
        sse_list.append(sse)

    plt.plot(range(2, len(sse_list) + 2), sse_list, marker='o')
    plt.ylabel('SSE', fontsize=22)
    plt.xlabel('K', fontsize=22)
    plt.show()
    """

    # clusterizzazione
    kmeans = KMeans(n_clusters=4, n_init=10, max_iter=1000)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    # grafico 2D
    """
    plt.scatter(cdf['E'], cdf['SaleRate'], c=kmeans.labels_, s=20)
    plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='*', c='k')
    plt.show()
    quit()
    """

    combos = list(combinations(attr_cluster, 3))

    for c in combos:
        c1, c2, c3 = c

        # Grafico 3D
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")

        ax.scatter3D(cdf[c1], cdf[c2], cdf[c3], c=kmeans.labels_, s=20)
        ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], s=200, marker='*', c='k')
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
        ax.set_zlabel(c3)
        plt.show()
    

    """
    from math import pi
 
    # number of variable
    N = len(cdf.columns)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    for i in range(0, len(centers)):
        angles = [n / float(N) * 2 * pi for n in range(N)]
        values = centers[i].tolist()
        values += values[:1]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(polar=True)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], cdf.columns, color='grey', size=8) 
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.show()
    """

    plt.figure(figsize=(8, 4))
    for i in range(0, len(centers)):
        plt.plot(centers[i], marker='o', label='Cluster %s' % i)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xticks(range(0, len(cdf.columns)), cdf.columns, fontsize=18)
    plt.legend(fontsize=20)
    plt.show()
    """
    sn.heatmap(cdf.drop('CustomerID', axis=1).corr(), annot=True)
    plt.show()
    plt.close()

    pd.plotting.scatter_matrix(cdf,figsize=(15,15))
    plt.show()
    plt.close()
    """