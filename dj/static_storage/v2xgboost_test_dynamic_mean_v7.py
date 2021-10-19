import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tqdm

TEST_SIZE_DAYS = 1

coin_files = [
    "v2_lagging_5jul_8may_2020_21_ada.hdf",
    "v2_lagging_5jul_8may_2020_21_eth.hdf",
    "v2_lagging_5jul_8may_2020_21_link.hdf",
    "v2_lagging_5jul_8may_2020_21_bnb.hdf",
]

DAY_LEN = 24


def simulate(shift_in_hours, path_to_hdf, train_hours, test_hours, coin, mean_const=1):
    sides = []
    df_orig = pd.read_hdf(path_to_hdf)
    df_orig = df_orig.fillna(0)
    del df_orig["trade_in_3h_usdtfutures_{}USDT_close_price".format(coin)]
    del df_orig["trade_in_2h_usdtfutures_{}USDT_close_price".format(coin)]

    start = 60 * (shift_in_hours - (train_hours + test_hours))
    end = 60 * (shift_in_hours)
    df_orig = df_orig[start:end]
    len(df_orig) / 60 / DAY_LEN

    df = df_orig.copy()
    Y = df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)]

    Y_MEAN = np.mean(np.abs(Y))
    Y_MEAN = Y_MEAN * mean_const

    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] > Y_MEAN] = 20
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < -Y_MEAN] = 10
    Y[df["trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)] < 9] = 1

    X = df_orig.loc[:, "usdtfutures_{}USDT_close_price".format(coin) :]

    test_size = train_hours / (train_hours + test_hours)
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

    assert len(Y_TEST_REAL) == len(Y_test)

    for i in range(len(Y_TEST_REAL)):
        v = 1
        if Y_test[i] == 10:
            v = -1
        if Y_test[i] == 1:
            continue
        assert v * Y_TEST_REAL[i] > 0

    initial_bank = 100
    rake = 0.00018

    trading_hours = 0
    skipped_hours = 0
    last_side = 0
    for i in range(1, len(Y_TEST_REAL), 60):
        sides.append(last_side)
        trading_hours += 1
        y_p = y_pred[i]
        y_r = Y_TEST_REAL[i]

        side = -1
        if y_p == 1:
            side = last_side
            """
            skipped_hours += 1
            if last_side != 0:
                rake_change = initial_bank * rake
                initial_bank = initial_bank - rake_change
            last_side = 0
            continue
            """

        if y_p == 20:
            side = 1

        profit = -1
        if y_r * side > 0:
            profit = 1

        change = initial_bank * abs(y_r) * profit
        initial_bank = initial_bank + change

        if side != last_side:
            rake_change = initial_bank * rake
            if side + last_side == 0:
                rake_change = rake_change * 2
            initial_bank = initial_bank - rake_change
        last_side = side

    return (
        initial_bank,
        trading_hours,
        skipped_hours / trading_hours,
        hours_in_test,
        model,
        sides,
    )


hours_in_test_lst = []
bank_results = []
file_to_pick_actual_result = "A"
picked = []
res_sides = []
hours_step = 6
#for shift_in_hours in tqdm.tqdm(range(20 * 24, 28 * 24, hours_step)):
for shift_in_hours in tqdm.tqdm(range(int(41000/60), int(44289/60), hours_step)):

    # these 4 are ["5 Jul, 2020", "22 Dec, 2020"]
    a = "v2_lagging_A22apr_22may_2021_eth.hdf"
    b = "v2_lagging_B22apr_22may_2021_eth.hdf"

    a2 = "v2_lagging_A22apr_22may_2021_ada.hdf"
    b2 = "v2_lagging_B22apr_22may_2021_ada.hdf"

    # these 4 are ["28 Apr, 2021", "28 May, 2021"]
    a = "v2_lagging_A28apr_28may_2021_eth.hdf"
    b = "v2_lagging_B28apr_28may_2021_eth.hdf"

    a2 = "v2_lagging_A28apr_28may_2021_ada.hdf"
    b2 = "v2_lagging_B28apr_28may_2021_ada.hdf"


    #a = "v2_raw_28apr_28may_2021_eth_raw.hdf"
    #b = "v2_raw_28apr_28may_2021_eth_raw.hdf"

    #a2 = "v2_raw_28apr_28may_2021_ada_raw.hdf"
    #b2 = "v2_raw_28apr_28may_2021_ada_raw.hdf"

    mean_const = 1 / 4

    train_hours, test_hours = 24, 24
    bank_a, trading_hours, skipped_ratio, hours_in_test, model, sides_a = simulate(
        shift_in_hours, a, train_hours, test_hours, "ETH", mean_const=mean_const
    )

    bank_a2, trading_hours, skipped_ratio, hours_in_test, model, sides_a2 = simulate(
        shift_in_hours, b, train_hours, test_hours, "ETH", mean_const=mean_const
    )

    bank_ax, trading_hours, skipped_ratio, hours_in_test, model, sides_ax = simulate(
        shift_in_hours, a2, train_hours, test_hours, "ADA", mean_const=mean_const
    )

    bank_a2x, trading_hours, skipped_ratio, hours_in_test, model, sides_a2x = simulate(
        shift_in_hours, b2, train_hours, test_hours, "ADA", mean_const=mean_const
    )

    train_hours, test_hours = 24, hours_step
    if file_to_pick_actual_result == "A":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            sides,
        ) = simulate(
            shift_in_hours, a, train_hours, test_hours, "ETH", mean_const=mean_const
        )
    if file_to_pick_actual_result == "A2":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            sides,
        ) = simulate(
            shift_in_hours, b, train_hours, test_hours, "ETH", mean_const=mean_const
        )

    if file_to_pick_actual_result == "Ax":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            sides,
        ) = simulate(
            shift_in_hours, a2, train_hours, test_hours, "ADA", mean_const=mean_const
        )
    if file_to_pick_actual_result == "A2x":
        (
            initial_bank,
            trading_hours,
            skipped_ratio,
            hours_in_test,
            model,
            sides,
        ) = simulate(
            shift_in_hours, b2, train_hours, test_hours, "ADA", mean_const=mean_const
        )

    banks = [
        bank_a,
        bank_a2,
        bank_ax,
        bank_a2x,
    ]
    max_i = np.argmax(banks)
    choices = [
        "A",
        "A2",
        "Ax",
        "A2x",
    ]
    assert len(banks) == len(choices)
    file_to_pick_actual_result = choices[max_i]
    picked.append(file_to_pick_actual_result)

    bank_results.append(initial_bank)
    res_sides.extend(sides)
    hours_in_test_lst.append(hours_in_test)

print(res_sides)
print(picked)
print(hours_in_test_lst)
print(bank_results)
print(np.mean(bank_results))
print(np.var(bank_results) / 1000000)
