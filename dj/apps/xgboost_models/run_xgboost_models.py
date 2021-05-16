import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tqdm
from apps.xgboost_models.create_hdfs_for_models import (
    create_A_hdf,
    create_B_hdf,
    create_C_hdf,
    create_D_hdf,
    create_E_hdf,
    create_F_hdf,
    create_A2_hdf,
    create_B2_hdf,
    create_C2_hdf,
    create_D2_hdf,
    create_E2_hdf,
    create_F2_hdf,
)

DAYS_EVAL = 2

model_codes = ["A", "B", "C", "D", "E", "F", "A2", "B2", "C2", "D2", "E2", "F2"]
hdf_create_functions = [
    create_A_hdf,
    create_B_hdf,
    create_C_hdf,
    create_D_hdf,
    create_E_hdf,
    create_F_hdf,
    create_A2_hdf,
    create_B2_hdf,
    create_C2_hdf,
    create_D2_hdf,
    create_E2_hdf,
    create_F2_hdf,
]
assert len(model_codes) == len(hdf_create_functions)


def get_X_from_df(df, coin):
    X = df.loc[:, "usdtfutures_{}USDT_close_price".format(coin) :]
    return X


def simulate(df_orig, coin):
    df_orig = df_orig.fillna(0)
    del df_orig["trade_in_3h_usdtfutures_{}USDT_close_price".format(coin)]
    del df_orig["trade_in_2h_usdtfutures_{}USDT_close_price".format(coin)]
    days = len(df_orig) / 60 / 24

    df = df_orig.copy()
    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]

    Y_MEAN = np.mean(np.abs(Y))
    Y_MEAN = Y_MEAN

    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] > Y_MEAN] = 20
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < -Y_MEAN] = 10
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < 9] = 1
    Y

    X = get_X_from_df(df_orig, coin)

    test_size = DAYS_EVAL / days
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=6, shuffle=False
    )
    hours_in_test = len(Y_test)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        booster="gbtree",
        gamma=0.1,
        sampling_method=["uniform", "gradient_based"][0],
        num_parallel_tree=1,
        reg_alpha=1,
        reg_lambda=0,
        eta=0.3,
    )
    model.fit(X_train, Y_train)
    print(model)

    y_pred = model.predict(X_test)

    Y_TEST_REAL = df_orig["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]
    Y_TEST_REAL = Y_TEST_REAL[-len(y_pred) :]
    Y_TEST_REAL

    assert len(Y_TEST_REAL) == len(Y_test)

    for i in range(len(Y_TEST_REAL)):
        v = 1
        if Y_test[i] == 10:
            v = -1
        if Y_test[i] == 1:
            continue
        assert v * Y_TEST_REAL[i] > 0

    initial_bank = 400000
    rake = 0.00018

    trading_hours = 0
    skipped_hours = 0
    last_side = -1
    for i in range(1, len(Y_TEST_REAL), 60):
        trading_hours += 1
        y_p = y_pred[i]
        y_r = Y_TEST_REAL[i]
        if y_p == 1:
            skipped_hours += 1
            continue

        side = -1
        if y_p == 20:
            side = 1

        profit = -1
        if y_r * side > 0:
            profit = 1

        change = initial_bank * abs(y_r) * profit
        initial_bank = initial_bank + change

        if side != last_side:
            rake_change = initial_bank * rake
            rake_change = rake_change * 2
            initial_bank = initial_bank - rake_change
        last_side = side

    return (
        initial_bank,
        trading_hours,
        skipped_hours / trading_hours,
        hours_in_test,
        model,
    )


def get_best_model_code():

    results = []
    for i in range(len(model_codes)):
        df, coin = hdf_create_functions[i]()
        (
            initial_bank,
            trading_hours,
            skipped_hours_ratio,
            hours_in_test,
            model,
        ) = simulate(df, coin)
        results.append(initial_bank)

    max_i = np.argmax(results)
    return model_codes[max_i], results[max_i]
