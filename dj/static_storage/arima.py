import pandas as pd
import pmdarima as pm
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import ndiffs


df_orig = pd.read_hdf("apr21a.hdf")
df_orig = df_orig.fillna(0)
#df_orig = df_orig[50000:70000]
REUSE_ORDER = 40


def evaluate_coin(coin):
    model_history_size = 60
    forecast_minutes = 400
    real_predicted_initial = []  # list of tuples
    aics = []
    ii = 0
    X_stationary = df_orig[
        "trade_in_1h_usdtfutures_{}USDT_close_price".format(coin)
    ].values
    X = df_orig["usdtfutures_{}USDT_close_price".format(coin)].values

    order = (1, 3, 1)
    reuse_order = 0

    for t in range(model_history_size*2, len(X) - forecast_minutes, forecast_minutes):
        ii += 1
        x = X[t - model_history_size : t]
        x_stationary = X_stationary[t - model_history_size : t]
        x = x_stationary
        initial = X[t]
        real = X[t + forecast_minutes]
        last = X[t - forecast_minutes]

        if not reuse_order and False:
            reuse_order = REUSE_ORDER
            # kpss_diffs = ndiffs(x, alpha=0.8, test="kpss", max_d=8)
            # adf_diffs = ndiffs(x, alpha=0.8, test="adf", max_d=8)
            # n_diffs = max(adf_diffs, kpss_diffs)
            start_i = int(max(model_history_size, t - model_history_size*10))
            x1 = X[start_i:t]
            # information_criterion, one of (‘aic’, ‘aicc’, ‘bic’, ‘hqic’, ‘oob’)
            model = pm.auto_arima(
                x1,
                # max_p=6,
                # max_d=6,
                # max_q=3,
                # start_q=1,
                method="lbfgs",
                trace=True,
                alpha=0.05,
                information_criterion="aic",
                seasonal=False,
                stationary=False,
            )
            order = model.get_params()["order"]

        reuse_order = reuse_order - 1
        #model = ARIMA(x[0:5], order=order)
        #model_fit = model.fit()
        #output = model_fit.forecast(steps=forecast_minutes)
        output = [1,2]
        stationary_predicted = sum(output)
        if stationary_predicted > 0:
            predicted = initial + 1
        else:
            predicted = initial - 1

        if initial > last:
            predicted = initial + 1
        else:
            predicted = initial - 1

        # predicted = output[-1]
        real_predicted_initial.append([real, predicted, initial])

    initial_bank = 400000
    rake = 0.00018
    last_side = 1
    for real, predicted, initial in real_predicted_initial:
        profit = -1
        if predicted > initial and real > initial:
            profit = 1
        if predicted < initial and real < initial:
            profit = 1

        if predicted > initial:
            side = 1
        else:
            side = -1

        change = abs((real / initial) - 1)

        initial_bank = initial_bank + (initial_bank * change * profit)

        if side != last_side:
            rake_change = initial_bank * rake
            rake_change = rake_change * 2
            initial_bank = initial_bank - rake_change

        last_side = side

    return (initial_bank, len(real_predicted_initial))


coin_banks = {}
futures = {}

with ProcessPoolExecutor(max_workers=5) as executor:
    ["ETH", "ADA", "XRP", "BTC"]
    for coin in ["ADA"]:
        futures[coin] = executor.submit(evaluate_coin, coin)

    for coin in futures:
        initial_bank, number_of_trades = futures[coin].result()
        coin_banks[coin] = (initial_bank, number_of_trades)

print(coin_banks)

s = []
for coin in coin_banks:
    s.append(coin_banks[coin][0])
avaraged_initial_bank = sum(s) / len(s)
print(avaraged_initial_bank)
