import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor(n_estimators=256, bootstrap=True, random_state=42, n_jobs=-1, verbose=1)


def rmsle(y_pred, y_true):
    return np.math.sqrt(np.mean(np.power(np.subtract(np.log(y_pred + 1), np.log(y_true.ravel() + 1)), 2)))


def cross_valid(train_DF, y_DF, test_size=0.4):
    X_train, X_test, y_train, y_test = train_test_split(train_DF, y_DF, test_size=test_size, random_state=42)
    y_train, y_test = np.array(y_train), np.array(y_test)
    model.fit(X_train, y_train.ravel())
    pred_X_train = model.predict(X_train)
    pred_X_test = model.predict(X_test)
    print('Error on X_train: ' + str(rmsle(pred_X_train, y_train)))
    print('Error on X_test:  ' + str(rmsle(pred_X_test, y_test)))


def train_and_test(train_DF, y_DF, test_DF):
    X_train = train_DF.values
    y_train = np.array(y_DF.values)
    X_test = test_DF.values
    model.fit(X_train, y_train.ravel())
    save_to_csv(test_DF, model.predict(X_test))


def save_to_csv(test_DF, Y_predict):
    filename = 'output/result.csv'
    test_DF['price_doc'] = Y_predict
    df = test_DF[['price_doc']]
    df.to_csv(filename, index=True)
