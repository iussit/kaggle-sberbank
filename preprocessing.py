import math
import pandas as pd


def missing_statistics(column):
    return {'Length': len(column), 'NA': column.isnull().sum(), '%': round(float(column.isnull().sum()) / len(column), 2)}


def print_missing_statistics(column):
    missing_st = missing_statistics(column)
    if int(missing_st['NA']) != 0:
        print(str(column.name) + ' missing statistics:')
        print('  Length: ' + str(missing_st['Length']))
        print('  NA:     ' + str(missing_st['NA']))
        print('  %:      ' + str(missing_st['%']))


def unique_statistics(column):
    from collections import Counter
    return Counter(column.dropna().values).items()


def print_unique_statistics(column):
    unique_st = unique_statistics(column)
    for key, value in unique_st:
        print('  ' + str(key) + ': ' + str(value))


def differences_statistics(first_column, second_column):
    return list(set(first_column.dropna().values) - set(second_column.dropna().values))


def print_differences_statistics(first_column, second_column):
    differences_st = differences_statistics(first_column, second_column)
    print('differences statistics between ' + str(first_column.name) + ' and ' + str(second_column.name) + ':')
    for elem in differences_st:
        print(str(elem))


def transform_to_one_hot(data_frame, feature, dummy_na=False):
    one_hot = pd.get_dummies(data_frame[feature], prefix=feature, dummy_na=dummy_na)
    data_frame = data_frame.drop(feature, axis=1)
    data_frame = data_frame.join(one_hot)
    return data_frame


def date_reductor(row_data):
    value = int(row_data)

    if 0 <= value < 17:
        value += 2000
    elif value == 20:
        value = 2000
    elif value == 71:
        value = 1971
    elif value == 215:
        value = 2015
    elif value == 4965:
        value = 1949
    elif value == 20052009:
        value = 2005

    row_data = value
    return row_data


def inf_to(data_frame, feature, fill_val=-1):
    data_frame[feature] = data_frame[feature].apply(lambda elem: fill_val if math.isinf(elem) else elem)
    return data_frame


def na_to_onehot(data_frame, feature, fill_val=-2):
    if data_frame[feature].isnull().values.any():
        data_frame[[feature + '_na_flag']] = data_frame[[feature]]. \
            apply(lambda x: na_flag_creator(x, feature), axis=1)

    if fill_val > -2:
        data_frame[[feature]] = data_frame[[feature]].fillna(fill_val)

    return data_frame


def na_flag_creator(row_data, column_name):
    row_data[column_name] = 1 if math.isnan(row_data[column_name]) else 0
    return row_data


def ordered_merge(data_frame, add_data_frame, on=None, fill_na=False):
    data_frame = pd.merge_ordered(data_frame, add_data_frame, on=on, how='left')
    if fill_na:
        data_frame = data_frame.fillna(data_frame.mean()[list(add_data_frame)])
    return data_frame
