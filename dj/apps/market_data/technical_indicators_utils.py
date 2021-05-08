import pandas as pd


def SMA22(df: pd.DataFrame, window: int, kline_attrs: list, shifts: int):
    for one in kline_attrs:
        nm = "SMA({}_{})".format(one, window)
        df[nm] = df[one].rolling(window).mean()
        for i in range(shifts):
            df["shift{}_".format(i + 1, nm)] = df[nm].shift(window * (i + 1))
    return df


def SMA(df: pd.DataFrame, window: int, kline_attrs: list, shifts: int):
    for one in kline_attrs:
        nm = "SMA({}_{})".format(one, window)
        df[nm] = df[one].rolling(window).mean()
        for i in range(shifts):
            df["shift{}_{}".format(i+1, nm)] = df[nm].shift(30)
    return df


# Acceleration Bands
def ABANDS():
    """
    Upper Band = Simple Moving Average (High * ( 1 + 4 * (High - Low) / (High + Low)))

    Middle Band = Simple Moving Average

    Lower Band = Simple Moving Average (Low * (1 - 4 * (High - Low)/ (High + Low)))

    :return:
    """
    pass
