"""
Custom functions for the analysis, putted in this separated file to lighten the notebook.
"""
from copy import deepcopy
import pandas as pd
import pickle


def read_dataset():
    """Read the dataset using Pandas, keeping only bought items."""
    df = pd.read_csv("../DM_25_TASK1/customer_supermarket_2.csv", index_col=0, parse_dates=["PurchaseDate"])
    return df[df['Qta'] > 0]


def remove_baskets(df, threshold):
    """Keep only customers with more than `threshold` baskets."""
    customers = df.groupby('CustomerID').agg({'BasketID': 'nunique'})
    customers = customers[customers >= threshold].dropna().index.values
    return df[df['CustomerID'].isin(customers)]


def sequentialize(df, return_times=False):
    """Convert a dataset into its sequential form. It can also return the time stamps of the baskets."""
    seq_data = []
    times = []
    for customer in df.groupby('CustomerID'):
        customer = customer[1]
        tmp = []
        tmp2 = []
        for basket in customer.groupby('BasketID'):
            basket = basket[1]
            purchases = list( basket['ProdID'].unique() )
            time = basket['PurchaseDate'].max()
            tmp.append(purchases)
            tmp2.append(time)
        seq_data.append(tmp)
        times.append(tmp2)
    if not return_times:
        return seq_data
    return seq_data, times


def read_write_result(read, min_baskets, min_sup, max_span=None, min_gap=None, max_gap=None):
    """Read/write a result_set from/to a pickle file.
    """
    # Build filename
    filename = f'gsp_res/{min_baskets}mb_{int(min_sup*100)}ms'
    if max_span:
        filename += f'_{str(max_span)}maxspan'
    if min_gap:
        filename += f'_{str(min_gap)}mingap'
    if max_gap:
        filename += f'_{str(max_gap)}maxgap'
    filename += '.pickle'

    if read:
        # Read GSP results
        with open(filename, 'rb') as handle:
            result_set = pickle.load(handle)

        # Sort by support
        result_set.sort(key=lambda x: x[1], reverse=True)

        return result_set
    else:
        # Write GSP results
        with open(filename, 'wb') as handle:
            pickle.dump(result_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_tuples_to_list(result_set):
    for i in range(len(result_set)):
        result_set[i] = list(result_set[i])
    return result_set


def print_distribution(result_set):
    """Compute distribution of the lengths of sequences and n. of sequences containing duplicates
    """
    cnt_len = {1:0, 2:0, 3:0, 4:0, 5:0}
    cnt_duplicates = 0
    for r in result_set:
        r = r[0]
        tmp = []
        for l in r:
            tmp.extend(l)
        len_tmp = len(tmp)
        cnt_len[len_tmp] += 1
        if len(set(tmp)) < len_tmp:
            cnt_duplicates += 1

    print(f"Distribution of lengths: {cnt_len}")
    print(f"Sequences containing duplicates: {cnt_duplicates} / {len(result_set)}")


def compute_patterns_mean_qta(result_set_original, df):
    """Compute the mean qta value over the transactions composing each pattern.
    """
    result_set = deepcopy(result_set_original)

    # Prepare result_set
    for i in range(len(result_set)):
        tmp = []
        # Create a nested list for future storage of mean qta of each item in the patterns
        for basket in result_set[i][0]:
            tmp2 = []
            for item in basket:
                tmp2.append(0)
            tmp.append(tmp2)
        result_set[i].append(tmp)

    """ PSEUDOCODICE
    per ogni cliente in clienti:
        per ogni risultato in risultati:
            per ogni carrello_1 in risultato: # carrello_res
                per ogni carrello_2 in cliente: # carrello_customer
                    carrello_1 è contenuto in carrello_2?
                        SI: scorri carrello_1 col prossimo carrello (fino a quando non finiscono, se finiscono allora questo cliente è ok)
                        NO: scorri carrello_2 col prossimo carrello (fino a quando non finiscono, se finiscono allora questo cliente NON è ok)
    """
    # Find original transactions for each pattern and collect statistics
    out = []
    n_customers = len(df['CustomerID'].unique())
    for customer_i, customer in enumerate(df.groupby('CustomerID')):
        # Progress
        print(f"{customer_i+1} / {n_customers}")
        # Extract baskets from the customer
        baskets_customer = list(enumerate([x[1] for x in customer[1].groupby('BasketID')]))
        
        for result_i in range(len(result_set)):
            res = result_set[result_i][0]

            # Compare the baskets in the result against those of the customer
            bc_i = 0
            transactions = []
            for basket_res in res:
                for i, basket_customer in baskets_customer[bc_i:]:
                    entries = basket_customer[basket_customer['ProdID'].isin(basket_res)]
                    entries = entries.groupby('ProdID').aggregate({'Qta': 'sum'})
                    if len(entries) >= len(basket_res):
                        bc_i = i + 1
                        transactions.append(entries)
                        break
                else: # We iterated over all the baskets of the customer without finding a match for basket_res
                    break
            else:
                # Compute qta for each item in the pattern
                for i, basket in enumerate(transactions):
                    for j in range(len(basket)):
                        item = basket.iloc[j]
                        result_set[result_i][2][i][j] += item['Qta']

                out.append(transactions)

    # Compute mean of the qta previously found
    for res in result_set:
        min_sup = res[1]
        sup = min_sup * n_customers
        for i in range(len(res[2])):
            for j in range(len(res[2][i])):
                res[2][i][j] = round(res[2][i][j] / sup)
    return result_set


def prodID_to_prodDescr(result_set, df):
    """Convert ProductIDs to readable descriptions.
    """
    for r_i, result in enumerate(result_set):
        tmp = []
        for b_i, basket in enumerate(result_set[r_i][0]):
            tmp2 = []
            for p_i, p in enumerate(basket):
                tmp2.append(df[df['ProdID'] == p]['ProdDescr'].iloc[0])
            tmp.append(tmp2)
        result_set[r_i][0] = tmp

