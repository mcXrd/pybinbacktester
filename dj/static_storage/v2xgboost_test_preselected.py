import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tqdm

TEST_SIZE_DAYS = 2

coin_files = [
    "v2_lagging_5jul_8may_2020_21_ada.hdf",
    "v2_lagging_5jul_8may_2020_21_eth.hdf",
    "v2_lagging_5jul_8may_2020_21_link.hdf",
    "v2_lagging_5jul_8may_2020_21_bnb.hdf",
]


def simulate(shift, path_to_hdf, days, coin, last_side):
    df_orig = pd.read_hdf(path_to_hdf)
    df_orig = df_orig.fillna(0)
    del df_orig["trade_in_3h_usdtfutures_{}USDT_close_price".format(coin)]
    del df_orig["trade_in_2h_usdtfutures_{}USDT_close_price".format(coin)]
    start = 60 * 24 * (shift)
    end = 60 * 24 * (days + shift)
    df_orig = df_orig[start:end]
    len(df_orig) / 60 / 24

    df = df_orig.copy()
    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]

    Y_MEAN = np.mean(np.abs(Y))
    Y_MEAN = Y_MEAN

    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] > Y_MEAN] = 20
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < -Y_MEAN] = 10
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < 9] = 1
    Y

    X = df_orig.loc[:, "usdtfutures_{}USDT_close_price".format(coin) :]

    test_size = TEST_SIZE_DAYS / days
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
    for i in range(1, len(Y_TEST_REAL), 60):
        trading_hours += 1
        y_p = y_pred[i]
        y_r = Y_TEST_REAL[i]

        side = -1
        if y_p == 1:
            skipped_hours += 1
            side = last_side

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
        last_side,
    )


hours_in_test_lst = []
bank_results = []
file_to_pick_actual_result = "A"
picked = []
picked = [
    "B2",
    "C2",
    "F2",
    "F",
    "C2",
    "E2",
    "E2",
    "D",
    "D2",
    "E2",
    "A",
    "A2",
    "A",
    "E",
    "E2",
    "E",
    "A",
    "F",
    "D2",
    "B",
    "A",
    "A2",
    "F",
    "F2",
    "D",
    "F",
    "F2",
    "D",
    "A",
    "A2",
    "A2",
    "D2",
    "B",
    "C2",
    "D2",
    "F2",
    "E2",
    "F2",
    "C2",
    "D2",
    "A2",
    "D2",
    "B2",
    "B2",
    "A2",
    "B2",
    "A2",
    "A2",
    "E2",
    "E",
    "E",
    "E2",
    "A2",
    "F2",
    "F",
    "E2",
    "E2",
    "E",
    "A2",
    "F",
    "E2",
    "F2",
    "F",
    "E",
    "F2",
    "D2",
    "D2",
    "A2",
    "B2",
    "A2",
    "A2",
    "A2",
    "F2",
    "D2",
    "A2",
    "E",
    "C2",
    "A2",
    "E",
    "D2",
    "F2",
    "D2",
    "E",
    "B2",
    "D2",
    "A2",
    "F2",
    "F",
    "E2",
    "D",
    "B2",
    "F",
    "B",
    "C2",
    "F2",
    "D",
    "F2",
    "A2",
    "A2",
    "F2",
    "D2",
    "F2",
    "F",
    "A2",
    "D2",
    "F2",
    "C",
    "B2",
    "B2",
    "F2",
    "C2",
    "D2",
    "C2",
    "F2",
    "C",
    "B2",
    "C2",
    "C2",
    "C",
    "D",
    "C",
    "A",
    "E2",
    "B2",
    "B2",
    "C2",
    "E",
    "B",
    "B",
    "A",
    "D",
    "B",
    "A",
    "A",
    "F",
    "B2",
    "E2",
    "D",
    "C2",
    "C2",
    "E2",
    "D2",
    "C2",
    "F",
    "B2",
    "F",
    "C2",
]
i = -1
for shift in tqdm.tqdm(range(1, 305 - 11, TEST_SIZE_DAYS)):
    i += 1
    if i == 0:
        continue
    a = "v2_lagging_A5jul_8may_2020_21_eth.hdf"
    b = "v2_lagging_B5jul_8may_2020_21_eth.hdf"
    c = "v2_lagging_C5jul_8may_2020_21_eth.hdf"

    a2 = "v2_lagging_A5jul_8may_2020_21_ada.hdf"
    b2 = "v2_lagging_B5jul_8may_2020_21_ada.hdf"
    c2 = "v2_lagging_C5jul_8may_2020_21_ada.hdf"

    code = picked[i - 1]
    last_side_eth = -1
    last_side_ada = -1
    if code == "A":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_eth,
        ) = simulate(shift, a, 5, "ETH", last_side_eth)
    if code == "B":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_eth,
        ) = simulate(shift, b, 5, "ETH", last_side_eth)
    if code == "C":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_eth,
        ) = simulate(shift, c, 5, "ETH", last_side_eth)
    if code == "D":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_eth,
        ) = simulate(shift, a, 11, "ETH", last_side_eth)
    if code == "E":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_eth,
        ) = simulate(shift, b, 11, "ETH", last_side_eth)
    if code == "F":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_eth,
        ) = simulate(shift, c, 11, "ETH", last_side_eth)
    if code == "A2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_ada,
        ) = simulate(shift, a2, 5, "ADA", last_side_ada)
    if code == "B2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_ada,
        ) = simulate(shift, b2, 5, "ADA", last_side_ada)
    if code == "C2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_ada,
        ) = simulate(shift, c2, 5, "ADA", last_side_ada)
    if code == "D2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_ada,
        ) = simulate(shift, a2, 11, "ADA", last_side_ada)
    if code == "E2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_ada,
        ) = simulate(shift, b2, 11, "ADA", last_side_ada)
    if code == "F2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            last_side_ada,
        ) = simulate(shift, c2, 11, "ADA", last_side_ada)

    bank_results.append(initial_bank)
    hours_in_test_lst.append(hours_in_test)

print(picked)
print(hours_in_test_lst)
print(bank_results)
print(np.mean(bank_results))
