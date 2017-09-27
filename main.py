from time import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

import preprocessing as prep
import learn_and_predict as lp

normal = [
    'full_sq', 'life_sq', 'kitch_sq', 'num_room', 'floor', 'max_floor',
    'kindergarten_km', 'swim_pool_km', 'theater_km', 'fitness_km', 'green_zone_km', 'park_km', 'water_km',
    'metro_km_walk', 'metro_km_avto', 'metro_min_walk', 'metro_min_avto', 'kremlin_km',
    'build_count_block', 'build_count_wood', 'build_count_frame', 'build_count_brick',
    'build_count_monolith', 'build_count_panel', 'build_count_foam', 'build_count_mix', 'build_count_slag'
]

categorical = [
    'sub_area', 'state', 'material', 'product_type'
]

special = [
    'build_year'
]

macro = [
    'usdrub', 'eurrub', 'brent',
    'balance_trade', 'balance_trade_growth', 'average_provision_of_build_contract',
    'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_rate', 'mortgage_value', 'mortgage_rate',
    'income_per_cap', 'rent_price_4+room_bus', 'apartment_build'
]


start_time = time()

print('Read CSV')
features = normal + categorical + special + ['id', 'timestamp']
train_DF = pd.read_csv('data/train.csv', parse_dates=['timestamp'], usecols=(features + ['price_doc']))
test_DF = pd.read_csv('data/test.csv', parse_dates=['timestamp'], usecols=features)
macro_DF = pd.read_csv('data/macro.csv', parse_dates=['timestamp'], usecols=(['timestamp'] + macro))


print('Merge Train/Test')
# train_DF.ix[train_DF[train_DF.full_sq == 0].index, 'full_sq'] = 50
# train_DF = train_DF[train_DF.price_doc / train_DF.full_sq <= 600000]
# train_DF = train_DF[train_DF.price_doc / train_DF.full_sq >= 10000]

train_DF_size = len(train_DF)
all_DF = pd.concat([train_DF, test_DF], axis=0)
# all_DF.ix[all_DF[all_DF.full_sq == 0].index, 'full_sq'] = 50

print('Preprocessing -->> Macro Features:')
all_DF = prep.ordered_merge(all_DF, macro_DF, on='timestamp', fill_na=True)
for index, feature in enumerate(macro):
    all_DF = prep.na_to_onehot(all_DF, feature, fill_val=-1)
    print('  ' + str(index + 1) + '/' + str(len(macro)) + '\t' + str(feature))

all_DF = all_DF.set_index(all_DF['id'])
all_DF = all_DF.drop('id', axis=1)

print('Preprocessing -->> Special Features:')

# all_DF.ix[all_DF[all_DF.kitch_sq >= all_DF.life_sq].index, "kitch_sq"] = np.NaN
# print('  1/13\t kitch_sq >= life_sq')
#
# all_DF.ix[all_DF[(all_DF.kitch_sq == 0).values + (all_DF.kitch_sq == 1).values].index, "kitch_sq"] = np.NaN
# print('  2/13\t kitch_sq == 0 or kitch_sq == 1')
#
# all_DF.ix[all_DF[all_DF.life_sq > all_DF.full_sq].index, ["life_sq", "full_sq"]] = np.NaN
# print('  3/13\t life_sq > full_sq')
#
# all_DF.ix[all_DF[all_DF.floor == 0].index, "floor"] = np.NaN
# print('  4/13\t floor == 0')
#
# all_DF.ix[all_DF[all_DF.max_floor == 0].index, "max_floor"] = np.NaN
# print('  5/13\t max_floor == 0')
#
# all_DF.ix[all_DF[all_DF.floor > all_DF.max_floor].index, "max_floor"] = np.NaN
# print('  6/13\t floor > max_floor')

all_DF = prep.na_to_onehot(all_DF, 'build_year', fill_val=-1)
all_DF['build_year'] = all_DF['build_year'].apply(prep.date_reductor)
print('  7/13\t build_year')

# all_DF['rel_floor'] = all_DF['floor'] / all_DF['max_floor'].astype(float)
# all_DF = prep.inf_to(all_DF, 'rel_floor', fill_val=-1)
# all_DF = prep.na_to_onehot(all_DF, 'rel_floor', fill_val=-1)
# print('  8/13\t rel_floor')
#
# all_DF['rel_kitch_sq'] = all_DF['kitch_sq'] / all_DF['full_sq'].astype(float)
# all_DF = prep.inf_to(all_DF, 'rel_floor', fill_val=-1)
# all_DF = prep.na_to_onehot(all_DF, 'rel_kitch_sq', fill_val=-1)
# print('  9/13\t rel_kitch_sq')
#
# month_year = (all_DF.timestamp.dt.month + all_DF.timestamp.dt.year * 100)
# month_year_cnt_map = month_year.value_counts().to_dict()
# all_DF['month_year_cnt'] = month_year.map(month_year_cnt_map)
# all_DF = prep.na_to_onehot(all_DF, 'month_year_cnt', fill_val=-1)
# print('  10/13\t month_year_cnt')
#
# week_year = (all_DF.timestamp.dt.weekofyear + all_DF.timestamp.dt.year * 100)
# week_year_cnt_map = week_year.value_counts().to_dict()
# all_DF['week_year_cnt'] = week_year.map(week_year_cnt_map)
# all_DF = prep.na_to_onehot(all_DF, 'week_year_cnt', fill_val=-1)
# print('  11/13\t week_year_cnt')
#
# all_DF['month'] = all_DF.timestamp.dt.month
# all_DF = prep.na_to_onehot(all_DF, 'month', fill_val=-1)
# print('  12/13\t month')
#
# all_DF['day_of_week'] = all_DF.timestamp.dt.dayofweek
# all_DF = prep.na_to_onehot(all_DF, 'day_of_week', fill_val=-1)
# print('  13/13\t day_of_week')

all_DF = all_DF.drop('timestamp', axis=1)

print('Preprocessing -->> Normal Features:')
for index, feature in enumerate(normal):
    all_DF = prep.na_to_onehot(all_DF, feature, fill_val=-1)
    print('  ' + str(index + 1) + '/' + str(len(normal)) + '\t' + str(feature))

print('Preprocessing -->> Categorical Features:')
if 'state' in categorical:
    all_DF['state'] = all_DF['state'].apply(lambda elem: elem if not np.isnan(elem) and int(elem) != 33 else 3)
for index, feature in enumerate(categorical):
    all_DF = prep.transform_to_one_hot(all_DF, feature, dummy_na=True)
    print('  ' + str(index + 1) + '/' + str(len(categorical)) + '\t' + str(feature))

print('Clustering')
cluster_DF = all_DF[['full_sq', 'life_sq', 'kitch_sq', 'num_room', 'floor', 'max_floor']]

cluster = MiniBatchKMeans(n_clusters=128, max_iter=1024, max_no_improvement=32, batch_size=64, n_init=16, random_state=42, verbose=1)
# cluster = DBSCAN(algorithm='auto', n_jobs=-1)
scaler = MinMaxScaler(feature_range=(0, 1))
cluster.fit(scaler.fit_transform(cluster_DF))
all_DF['clustering'] = cluster.labels_
# all_DF = prep.transform_to_one_hot(all_DF, 'clustering')

print('Split Train/Test')
train_DF = all_DF[:train_DF_size]
test_DF = all_DF[train_DF_size:]
test_DF = test_DF.drop('price_doc', axis=1)

preprocessing_time = int(time() - start_time)
print('\nPreprocessing Time: ' + str(int(preprocessing_time / 60)) + 'm ' + str(preprocessing_time % 60) + 's\n')


print('Cross Valid')
train_X_DF = train_DF.drop('price_doc', axis=1)
train_Y_DF = train_DF[['price_doc']]
lp.cross_valid(train_X_DF, train_Y_DF, test_size=0.4)

print('Train and Test')
test_X_DF = test_DF
test_X_DF = test_X_DF.fillna(-1, axis=0)
lp.train_and_test(train_X_DF, train_Y_DF, test_X_DF)

runtime = int(time() - start_time)
print('\nRuntime: ' + str(int(runtime / 60)) + 'm ' + str(runtime % 60) + 's\n')
