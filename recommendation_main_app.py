from funcs import *


# 4 DIFFERENT APPROACHES WERE USED IN THIS PROJECT
    # 1) ASSOCIATION RULES
    # 2) USER BASED COLLABORATIVE FILTERING
    # 3) POPULARITY BASED RECOMMENDATION
    # 4) ITEM BASED COLLABORATIVE FILTERING


# PRODUCT EXAMPLES
# product_id_1 = "HBV00000JUHKU"
# product_id_2 = "HBV00000U2B18"
# product_id_4 = "HBV00000NFHXT"
# product_id_5 = "OFIS3101-080"


def main():
    df = create_row_dataframe ()

    df_prep = read_data_prepared (dataframe=df)

    session_pro_df = read_session_pro_df (dataframe=df_prep)

    rules = read_rules_df (session_product_df=session_pro_df)

    user_product_matrix = read_user_product_matrix_df (prep_data=df_prep)

    my_cart = Cart ()

    try:
        while True:
            product_id = str (input (
                "Sepete Eklemek İstediğiniz Ürün Kodunu Giriniz. Alışverişinizi tamamladıysanız 'e', sepetinizi temizlemek için 'c' tuşuna basabilirsiniz.\nSepetinizi görüntülemek için 'p' tuşuna basabilirsiniz.\n"))
            if product_id.lower () == "e":
                print (my_cart.shopping_list)
                break
            elif product_id.lower () == "c":
                my_cart.clear_cart ()
            elif product_id.lower () == "p":
                my_cart.display_cart ()
            elif (product_id in df_prep["PRODUCTID"].unique ()):
                print ("Sepete eklenen ürün: {}".format (product_name (df_prep, product_id)))
                print ("Ürün önerileriniz yükleniyor... Biraz zaman alabilir..")

                my_cart.add_to_cart (product_id)
                best_same_diff_cat_pro_list = bestseller_same_diff_cat_day_time (df_prep, product_id)
                arl_recommender_list = arl_recommender (rules, product_id, 5)
                user_based_recommender_list = user_based_recommendation (session_df=session_pro_df, prep_df=df_prep,
                                                                         shopping_cart=my_cart.shopping_list)
                item_based_recommender_list = item_based_recommendation (user_product_matrix, product_id)

                recommendation_list = best_same_diff_cat_pro_list + arl_recommender_list + user_based_recommender_list + item_based_recommender_list
                recommendations = list (set (recommendation_list))

                final_recommendations = random.sample (recommendations, 10)

                print ("#" * 50)
                for product in final_recommendations:
                    print (product_name (df_prep, product))
                print ("#" * 50)
            else:
                print ("Girilen ürün kodu hatalı. Lütfen tekrar deneyiniz.\n")
    except ValueError:
        print ("Önerilecek ürün bulunamadı.")


if __name__ == '__main__':
    main ()
