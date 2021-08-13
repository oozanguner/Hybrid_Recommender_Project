from funcs import *

# YOU SHOULD RUN THIS FILE BEFORE YOU START THE PROJECT, FOR DOWNLOADING THE DATAFRAMES TO YOUR LOCAL WHICH USED IN THIS PROJECT.

df = create_row_dataframe ()

df_prep = read_data_prepared (df, True)

session_pro_df = read_session_pro_df (dataframe=df_prep, upgrade=True)

rules = read_rules_df (session_product_df=session_pro_df, upgrade=True)

user_product_matrix = read_user_product_matrix_df (prep_data=df_prep, upgrade=True)