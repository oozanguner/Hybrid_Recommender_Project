from funcs import *

df = create_row_dataframe ()

df_prep = data_preparation (df)


def plot_most_purch_cats(count=3):
    """
    Plots top specified number purchased product categories of all dataset with 'count' parameter according to day and time.
    """
    df_prep = pd.read_pickle ("df_prep.pickle")

    category_df = df_prep.groupby ("NEW_DAY_TIME").agg ({"CATEGORY": "value_counts"})
    category_df.rename (columns={"CATEGORY": "COUNT"}, inplace=True)
    category_df = category_df.reset_index ()

    top_categories_all_time = category_df.groupby ("CATEGORY").agg ({"COUNT": "sum"}).sort_values (by="COUNT",
                                                                                                   ascending=False)

    top_count_categories_list = top_categories_all_time.head (count).index

    best_sales_time_of_top_count_cats = category_df.loc[category_df["CATEGORY"].isin (top_count_categories_list)] \
        .sort_values (by="COUNT", ascending=False)

    plt.figure (figsize=(40, 20))
    sns.barplot (data=best_sales_time_of_top_count_cats, x="NEW_DAY_TIME", y="COUNT", hue="CATEGORY")
    plt.legend (loc=1, prop={'size': 30})
    plt.ylabel ("COUNT", fontsize=20)
    plt.xticks (rotation=50, fontsize=20)
    plt.show ();


plot_most_purch_cats ()
