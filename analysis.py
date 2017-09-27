import pandas as pd
import numpy as np

import preprocessing as prep

print('Read CSV')
train_DF = pd.read_csv('data/train.csv', parse_dates=['timestamp']).drop(['id', 'price_doc'], axis=1)
test_DF = pd.read_csv('data/test.csv', parse_dates=['timestamp']).drop(['id'], axis=1)
macro_DF = pd.read_csv('data/macro.csv', parse_dates=['timestamp'])

df_all = pd.concat([train_DF, test_DF], axis=0)

# equal_index = [601, 1896, 2791]
# bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
# print(df_all.loc[bad_index]['num_room'])

print(set(df_all.timestamp.dt.year))
print(set(df_all.timestamp.dt.month))
print(set(df_all.timestamp.dt.day))

# df_all.product_type.value_counts(normalize=True)
# df_all.floor.describe(percentiles=[0.9999])


# print(df_all[['full_sq', 'kitch_sq', 'life_sq']][df_all.kitch_sq > 1500])
#
# df_all = df_all.dropna()[:10]
# print(df_all[['product_type']])
# df_all.product_type.value_counts(normalize=True)
# print(df_all[['product_type']])


# all_DF.loc[all_DF.full_sq == 0, 'full_sq'] = 50
# all_DF = all_DF[all_DF.price_doc/all_DF.full_sq <= 600000]
# all_DF = all_DF[all_DF.price_doc/all_DF.full_sq >= 10000]

# prep.print_unique_statistics(all_DF['num_room'])

# print(all_DF[all_DF.floor == 0][['floor', 'max_floor']])
# print(all_DF[all_DF.kitch_sq >= all_DF.life_sq][['full_sq', 'kitch_sq', 'life_sq']])
# print(all_DF[(all_DF.kitch_sq == 0).values + (all_DF.kitch_sq == 1).values][['full_sq', 'kitch_sq']])
# print(all_DF[all_DF.kitch_sq > all_DF.full_sq][['full_sq', 'kitch_sq']])





# cols = ['build_year']
# for col in cols:
#     prep.print_unique_statistics(all_DF[col])

# macro_cols = [
#     'balance_trade', 'balance_trade_growth', 'eurrub', 'average_provision_of_build_contract',
#     'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_rate', 'mortgage_value', 'mortgage_rate',
#     'income_per_cap', 'rent_price_4+room_bus', 'museum_visitis_per_100_cap', 'apartment_build'
# ]
#
# for col in macro_cols:
#     print(prep.missing_statistics(macro_DF[col]))
#     print('\n')

# fucking_comma_cols = ['old_education_build_share', 'modern_education_share']
# for col in fucking_comma_cols:
#     macro_DF[col] = macro_DF[col].apply(lambda elem: float(str(elem).replace(',', '.')) if str(elem).find(',') != -1 else elem)


