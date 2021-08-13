import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import warnings
import random
import time
import sys

warnings.filterwarnings ("ignore")

pd.set_option ('display.max_rows', 30)
pd.set_option ('display.width', 500)
pd.set_option ('display.max_columns', 50)

from mlxtend.frequent_patterns import apriori, association_rules


class Cart:
    """
    Creates the Cart class.
    """

    def __init__(self):
        self.shopping_list = []

    def display_cart(self):
        """
        Displays the cart.
        """
        print (self.shopping_list)

    def add_to_cart(self, product_id):
        """
        Adds product to cart.
        """
        self.shopping_list.append (product_id)

    def clear_cart(self):
        """
        Clears the cart.
        """
        self.shopping_list.clear ()


def convert_json_to_df(path, recordPath):
    """
    Converts the json file to dataframe.
    """
    import json
    with open (path, 'r') as f:
        data = json.loads (f.read ())

    dataframe = pd.json_normalize (data, record_path=[recordPath])
    return dataframe


def check_df(dataframe, head=5):
    """
    Displays descriptive statistics, missing values, data size, data types and first rows of given dataframe as a parameter.
    """

    print ("##### SHAPE #####")
    print (dataframe.shape)
    print ("##### DTYPES #####")
    print (dataframe.dtypes)
    print ("##### MISSING VALUES #####")
    print (dataframe.isnull ().sum ())
    print ("##### DESCRIPTIVE STATISTICS #####")
    print (dataframe.describe ([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print ("##### FIRST " + str (head) + " ROWS #####")
    print (dataframe.head ())


def is_float(dataframe, col):
    """
    Returns True if the type of the given column is float if not returns False.
    """
    try:
        dataframe[col].astype (float)
        return True
    except ValueError:
        return False


def is_integer(dataframe, col):
    """
    Returns True if the type of the given column is integer if not returns False.
    """
    try:
        dataframe[col].astype (int)
        return True
    except ValueError:
        return False


def is_date(dataframe, col):
    """
     Returns True if the type of the given column is datetime if not returns False.
    """
    try:
        pd.to_datetime (dataframe[col])
        return True
    except ValueError:
        return False


def grab_col_types(dataframe):
    """
    Grabs the type of columns as float, integer, datetime or object and assigns it to grabbed version by converting.
    """
    for col in dataframe.columns:
        if is_float (dataframe, col):
            dataframe[col] = dataframe[col].astype (float)
        elif is_integer (dataframe, col):
            dataframe[col] = dataframe[col].astype (int)
        elif is_date (dataframe, col):
            dataframe[col] = pd.to_datetime (dataframe[col])
        else:
            dataframe[col] = dataframe[col].astype ("O")


def missing_col_ratio(dataframe, threshold=0.1):
    """
    Returns missing value counts and their missing value ratio of all columns.
    It also grabs the columns as a list which has missing values ratio that is higher than threshold value.
    """
    na_ratio_df = pd.DataFrame ({"NA_RATIO": dataframe.isnull ().sum () / len (dataframe),
                                 "NA_COUNT": dataframe.isnull ().sum ()})

    na_cols = na_ratio_df.loc[na_ratio_df["NA_RATIO"] > threshold].index.tolist ()

    return na_cols, na_ratio_df


def create_row_dataframe(
        events_path='/Users/ozanguner/PycharmProjects/Hybrid_Recommender/raw_datasets/events.json',
        events_record_path='events',
        meta_path='/Users/ozanguner/PycharmProjects/Hybrid_Recommender/raw_datasets/meta.json',
        meta_record_path='meta'):
    """
    Converts the json files include the function to dataframe.
    Then converts the columns types of 'events' and 'meta' dataframes to grabbed versions and merges
    'events_df' and 'meta_df' as a dataframe.

    Notes: Make sure the events json path and meta json path are in the right direction
    when you give these paths as parameter.
    """
    try:
        events_df = convert_json_to_df (events_path, events_record_path)
        meta_df = convert_json_to_df (meta_path, meta_record_path)

        grab_col_types (events_df)
        grab_col_types (meta_df)

        dataframe = events_df.merge (meta_df, how="left", on="productid")

        return dataframe
    except:
        print ("File Not Found. Please check the events json path and meta json path are in the right direction on your local when you give these paths as parameter.")




def data_preparation(dataframe):
    """
    Returns row dataframe to prepared dataframe by applying preprocessing.
    """
    dataframe.columns = [col.upper () for col in dataframe.columns]

    df_prep = dataframe.copy ()
    # Deleting the feature that has only one class.
    one_class_feature = [col for col in df_prep.columns if df_prep[col].nunique () == 1]
    df_prep.drop (one_class_feature, axis=1, inplace=True)

    na_cols_high_rated, missing_df = missing_col_ratio (df_prep)

    for col in na_cols_high_rated:
        df_prep[col].fillna ("None", inplace=True)

    df_prep.dropna (axis=0, inplace=True)

    # Feature Engineering.
    df_prep["NEW_EVENTTIME"] = pd.to_datetime (df_prep["EVENTTIME"].dt.strftime ('%Y-%m-%d'))
    df_prep["NEW_EVENTHOURS"] = df_prep["EVENTTIME"].dt.hour
    df_prep["NEW_WEEKDAY"] = df_prep["NEW_EVENTTIME"].dt.weekday
    day_map = {0: "MONDAY",
               1: "TUESDAY",
               2: "WEDNESDAY",
               3: "THURSDAY",
               4: "FRIDAY",
               5: "SATURDAY",
               6: "SUNDAY"}

    df_prep["NEW_WEEKDAY"] = df_prep["NEW_WEEKDAY"].map (day_map)

    df_prep["NEW_EVENTHOURS_RANGE"] = pd.cut (df_prep["NEW_EVENTHOURS"],
                                              bins=[df_prep["NEW_EVENTHOURS"].min () - 1, 3, 7, 11, 15, 19,
                                                    df_prep["NEW_EVENTHOURS"].max ()],
                                              labels=["0_3", "4_7", "8_11", "12_15", "16_19", "20_23"])

    df_prep["NEW_DAY_TIME"] = ["_".join (row) for row in df_prep[["NEW_WEEKDAY", "NEW_EVENTHOURS_RANGE"]].values]

    return df_prep


def read_data_prepared(dataframe, upgrade=False):
    """
    Converts row dataframe to prepared dataframe if 'upgrade' parameter is True or
    reads prepared dataframe from its pickle format that is already exist if 'upgrade' parameter is False.
    """
    if upgrade:
        df_prep = data_preparation (dataframe)
        df_prep.to_pickle ("df_prep.pickle")
    else:
        df_prep = pd.read_pickle ("df_prep.pickle")

    return df_prep


## ASSOCIATION RULES
def create_rules(session_pro_df, metric_name="support", minimum_support=0.002, minimum_threshold=0.002):
    """
    Returns dataframe created with products' association rules.
    """
    buy_diff_pro = session_pro_df.transpose ().sum ().sort_values (ascending=False).reset_index ()
    buy_diff_pro.rename (columns={0: "COUNT"}, inplace=True)

    # Selecting of session id that purchased more than 1 kind of product.
    sessions_more_than_one_products = buy_diff_pro.loc[buy_diff_pro["COUNT"] > 1, "SESSIONID"].tolist ()

    fin_pro_df = session_pro_df.loc[session_pro_df.index.isin (sessions_more_than_one_products)]

    # Apriori & Association Rules.
    freq_pro_sets = apriori (fin_pro_df, min_support=minimum_support, use_colnames=True, low_memory=True)

    rules = association_rules (freq_pro_sets, metric=metric_name, min_threshold=minimum_threshold)

    sorted_rules = rules.sort_values (metric_name, ascending=False)

    return sorted_rules


def read_rules_df(session_product_df, metric="support", upgrade=False):
    """
    Converts session id and product matrix to products' association rules dataframe if 'upgrade' parameter is True or
    reads products' association rules dataframe from its pickle format that is already exist if 'upgrade' parameter is False.
    """
    if upgrade:
        rules = create_rules (session_product_df, metric_name=metric)
        rules.to_pickle ("rules_df.pickle")
    else:
        rules = pd.read_pickle ("rules_df.pickle")
        rules = rules.sort_values (by=metric, ascending=False)

    return rules


def arl_recommender(rules_df, product_id, rec_count=10):
    """
    Converts given dataframe, which is created by association rules,
     to product recommendation list that includes given number of products
     as 'rec_count' parameter for given product id.
    """
    rec_list = list (rules_df.loc[rules_df["antecedents"].apply (lambda x: product_id in x), "consequents"])[
               0:20]
    recommendation_list = list (set ([list (x)[0] for x in rec_list]))[0:rec_count]

    return recommendation_list


def product_name(dataframe, productid):
    """
    Returns the name of the product whose id is given.
    """
    product = dataframe[dataframe["PRODUCTID"] == productid][["BRAND", "CATEGORY", "NAME", "PRODUCTID"]].values[
        0].tolist ()
    if product[0] == "None":
        product_name = ", ".join (product[1:])
    else:
        product_name = ", ".join (product)
    return product_name


## POPULARITY-BASED
def create_current_time(dataframe):
    """
    Creates day and time label from related dataframe columns includes current day and time by using datetime module.
    """
    current_hour = dt.datetime.now ().hour
    current_day_index = dt.datetime.now ().weekday ()
    day_map = {0: "MONDAY",
               1: "TUESDAY",
               2: "WEDNESDAY",
               3: "THURSDAY",
               4: "FRIDAY",
               5: "SATURDAY",
               6: "SUNDAY"}
    current_day_name = day_map[current_day_index]
    new_eventhours_range = dataframe.loc[dataframe["NEW_EVENTHOURS"] == current_hour, "NEW_EVENTHOURS_RANGE"].values[0]
    day_time_label = "_".join ([current_day_name, new_eventhours_range])
    return day_time_label


def bestseller_same_diff_cat_day_time(dataframe, product_id, diff_cat_rec_count=5, same_cat_rec_count=3):
    """
    According to the day and time of shopping, suggesting the most sold products in the same and different categories as the product added to the cart.
    The 'diff_cat_rec_count' parameter determines how many different categories of products will be recommended.
    The 'same_cat_rec_count' parameter determines how many different products will be recommended in the same category.
    """

    daytime_label = create_current_time (dataframe)

    agg_df = dataframe.groupby (["NEW_DAY_TIME", "CATEGORY", "PRODUCTID"]).agg ({"PRODUCTID": "count"})
    agg_df.rename (columns={"PRODUCTID": "COUNT"}, inplace=True)
    product_sales_df = agg_df.reset_index ().sort_values (by=["NEW_DAY_TIME", "CATEGORY", "COUNT"], ascending=False)

    # The category of product added to cart.
    category_purchased = product_sales_df.loc[product_sales_df["PRODUCTID"] == product_id, "CATEGORY"].values[0]

    # The whole categories.
    all_categories = product_sales_df["CATEGORY"].unique ()

    # Creating the recommendation list which includes the different categories, according to current day and time label.
    diff_bestseller_categories = [categories for categories in all_categories if categories != category_purchased]
    diff_category_product_rec = product_sales_df.loc[(product_sales_df["NEW_DAY_TIME"] == daytime_label)
                                                     & (product_sales_df["CATEGORY"]
                                                        .isin (diff_bestseller_categories))].groupby ("CATEGORY").head (
        1).sort_values (by="COUNT", ascending=False)["PRODUCTID"].tolist ()[:diff_cat_rec_count]

    # Creating the recommendation list which includes different products in the same categories with product added to cart, according to day and time label.
    same_category_all_products = product_sales_df.loc[(product_sales_df["NEW_DAY_TIME"] == daytime_label)
                                                      & (product_sales_df[
                                                             "CATEGORY"] == category_purchased), "PRODUCTID"].unique ()

    same_category_product_rec = [product for product in same_category_all_products if product != product_id][
                                :same_cat_rec_count]

    recommandation_list = diff_category_product_rec + same_category_product_rec

    return recommandation_list


## USER-BASED
def create_sessionid_product_matrix(dataframe):
    """
    Creates session id and product matrix for user-based collaborative recommendation.
    Focuses on whether the relevant product is in the session id or not.
    """
    session_pro_df = dataframe.groupby (['SESSIONID', 'PRODUCTID'])['PRODUCTID'].count ().unstack ().fillna (
        0).applymap (
        lambda x: 1 if x > 0 else 0)

    return session_pro_df


def read_session_pro_df(dataframe, upgrade=False):
    """
    Converts given prepared dataframe to session id-product matrix if 'upgrade' parameter is True or
    reads session id-product matrix from its pickle format that is already exist if 'upgrade' parameter is False.
    """
    if upgrade:
        session_pro_df = create_sessionid_product_matrix (dataframe)
        session_pro_df.to_pickle ("session_pro_df.pickle")
    else:
        session_pro_df = pd.read_pickle ("session_pro_df.pickle")

    return session_pro_df


def user_based_recommendation(session_df, prep_df, shopping_cart=[], rec_count=5):
    """
    Returns recommendation list which is created by using user based collaborative filtering approach.
    The 'shopping_cart' parameter represents the cart that is include added products.
    The 'rec_count' parameter determines maximum number of product will be recommended.
    """
    new_user_df = pd.DataFrame (index=["new_user"], columns=session_df.columns).fillna (0)
    my_cart_unique_list = list (set (shopping_cart))
    my_cart_unique_list = [col for col in my_cart_unique_list if col in session_df.columns]

    for product in my_cart_unique_list:
        new_user_df[product] = 1

    # Finding the similar session id with the new created session id.
    similar_user = session_df[my_cart_unique_list].T.sum ().sort_values (ascending=False).index[0]

    similar_user_products_df = prep_df.loc[
        prep_df["SESSIONID"] == similar_user, "PRODUCTID"].value_counts ().sort_values (ascending=False)

    recommendation_user_based = [product for product in similar_user_products_df.index if
                                 product not in my_cart_unique_list][:rec_count]

    return recommendation_user_based


## ITEM-BASED
def create_user_product_matrix_item_based(prep_data):
    """
    Creates session id-product matrix for item based collaborative recommendation.
    Focuses on how many of the relevant product is in the session ID.
    """
    user_product = prep_data.groupby ("SESSIONID").agg ({"PRODUCTID": "value_counts"}) \
        .rename (columns={"PRODUCTID": "COUNT"}) \
        .reset_index () \
        .sort_values (by=["SESSIONID", "COUNT"], ascending=False)

    user_product_matrix = pd.pivot_table (data=user_product, index="SESSIONID", columns="PRODUCTID", values="COUNT")

    return user_product_matrix


def read_user_product_matrix_df(prep_data, upgrade=False):
    """
    Converts given prepared dataframe to user-product matrix if 'upgrade' parameter is True or
    reads user-product matrix from its pickle format that is already exist if 'upgrade' parameter is False.
    """
    if upgrade:
        user_product_matrix = create_user_product_matrix_item_based (prep_data)
        user_product_matrix.to_pickle ("user_product_matrix.pickle")
    else:
        user_product_matrix = pd.read_pickle ("user_product_matrix.pickle")

    return user_product_matrix


def item_based_recommendation(user_pro_matrix, product_id, threshold=0.5, rec_count=4):
    """
    Returns recommendation list which is created by using item based collaborative filtering approach.
    The 'threshold' parameter represents the threshold for correlation of products.
    The 'rec_count' parameter determines maximum number of product will be recommended.
    """
    user_pro_matrix[product_id].sort_values (ascending=False)
    try:
        product_correlated = user_pro_matrix.corrwith (user_pro_matrix[product_id]).sort_values (ascending=False)

        # Removing the product added to cart from recommendation list.
        product_correlated = product_correlated.loc[product_correlated.index != product_id]

        # Finding the correlation of products.
        product_correlated.dropna (inplace=True)
        product_correlated = product_correlated.reset_index ()
        product_correlated.rename (columns={0: "CORR"}, inplace=True)

        item_based_rec_df = product_correlated[product_correlated["CORR"] > threshold].head (5)

        item_based_recommendation_list = random.sample (set (item_based_rec_df["PRODUCTID"]), rec_count)

    except ValueError:
        # item_based_recommendation_list = []
        item_based_recommendation_list = random.sample (set (item_based_rec_df["PRODUCTID"]), len (item_based_rec_df))

    return item_based_recommendation_list
