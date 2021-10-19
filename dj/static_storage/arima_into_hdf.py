import pandas as pd
import pmdarima as pm

coin = "ADA"
df_orig = pd.read_hdf("v2_lagging_5jul_8may_2020_21_ada.hdf")

X = df_orig["usdtfutures_{}USDT_close_price".format(coin)].values
model_history_size = 60
forecast_minutes = 60


for i in range(model_history_size * 2, len(X) - forecast_minutes, forecast_minutes):
    x = X[i : i + model_history_size]
    model = pm.auto_arima(
        x,
        # max_p=6,
        # max_d=6,
        # max_q=3,
        # start_q=1,
        method="lbfgs",
        trace=True,
        alpha=0.05,
        information_criterion="aic",
        seasonal=False,
        stationary=True,
    )
    output = model.predict(n_periods=forecast_minutes)

    X.iloc[i: i + model_history_size].loc[:,"newcol"]=3
    raise Exception(sum(output))
