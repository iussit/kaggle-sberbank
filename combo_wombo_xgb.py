import collections
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing

macro_cols = [
    'usdrub', 'eurrub', 'brent',
    'balance_trade', 'balance_trade_growth', 'average_provision_of_build_contract',
    'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_rate', 'mortgage_value', 'mortgage_rate',
    'income_per_cap', 'rent_price_4+room_bus', 'apartment_build'
]

print('Read data')
df_train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
df_test = pd.read_csv('data/test.csv', parse_dates=['timestamp'])
df_macro = pd.read_csv('data/macro.csv', parse_dates=['timestamp'])  # , usecols=['timestamp'] + macro_cols)

y_train = np.log1p(df_train['price_doc'].values)
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)

print('Add macro')
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')

print('Clear data')
df_all.life_sq[df_all.life_sq > df_all.full_sq] = np.NaN
df_all.life_sq[df_all.life_sq > df_all.full_sq * 10] = df_all.full_sq[df_all.life_sq > df_all.full_sq * 10]
df_all.life_sq[df_all.life_sq > 999] /= 100
df_all.life_sq[(df_all.life_sq > 100).values & (df_all.full_sq < 100).values] /= 10
df_all.life_sq[df_all.life_sq < 5] = np.NaN
df_all.full_sq[df_all.full_sq < 5] = np.NaN
df_all.build_year[df_all.kitch_sq > 1500] = df_all.kitch_sq[df_all.kitch_sq > 1500]
df_all.kitch_sq[df_all.kitch_sq >= df_all.life_sq] = np.NaN
df_all.kitch_sq[(df_all.kitch_sq == 0).values + (df_all.kitch_sq == 1).values] = np.NaN
df_all.full_sq[(df_all.full_sq > 150).values & (df_all.life_sq / df_all.full_sq < 0.3).values] = np.NaN
df_all.full_sq[df_all.life_sq > 200] = np.NaN
df_all.life_sq[df_all.life_sq > 200] = np.NaN
df_all.build_year[(df_all.build_year < 1500).values | (df_all.build_year > 4000).values] = np.NaN
df_all.num_room[(df_all.num_room == 0).values | (df_all.num_room >= 9).values] = np.NaN
df_all.floor[df_all.floor == 0] = np.NaN
df_all.max_floor[df_all.max_floor == 0] = np.NaN
df_all.max_floor[df_all.floor > df_all.max_floor] = np.NaN
df_all.state[df_all.state == 33] = np.NaN

print('Add features')
month_year = (df_all.timestamp.dt.year * 10000 + df_all.timestamp.dt.month * 100 + df_all.timestamp.dt.day)
month_year_cnt_map = month_year.value_counts().to_dict()

for year in range(2011, 2016):
    for month in range(1, 12):
        days = {1: 31, 2: 29 if (year % 4 == 0 and year % 100 != 0 or year % 400 == 0) else 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        for day in range(1, days[month]):
            if (year * 10000 + month * 100 + day) not in month_year_cnt_map:
                month_year_cnt_map[(year * 10000 + month * 100 + day)] = 0

month_year_cnt_map_sorted = collections.OrderedDict(sorted(month_year_cnt_map.items()))
val = list(month_year_cnt_map_sorted.values())

window = 30
res_val = []
for i in range(0, len(val)):
    sum = 0
    for j in range(i, min(i + window, len(val) - 1)):
        sum += val[j]
    res_val.append(sum)

index = 0
for k, v in month_year_cnt_map_sorted.items():
    month_year_cnt_map_sorted[k] = res_val[index]
    index += 1

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map_sorted)

# week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
# week_year_cnt_map = week_year.value_counts().to_dict()
# df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

df_all.apartment_name = df_all.sub_area + df_all['metro_km_avto'].astype(str)
df_all['room_size'] = df_all['life_sq'] / df_all['num_room'].astype(float)

df_all.drop(['timestamp'], axis=1, inplace=True)

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values))
        df_all[c] = lbl.transform(list(df_all[c].values))

x_train = df_all[:num_train]
x_test = df_all[num_train:]

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=20, show_stdv=False)
num_boost_rounds = len(cv_output)  # 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = np.exp(model.predict(dtest)) - 1

filename = 'output/xgb_result.csv'
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('output/combo_wombo_xgb.csv', index=False)
