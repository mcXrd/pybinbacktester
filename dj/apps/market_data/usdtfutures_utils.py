import time
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from typing import List, Callable
from binance_f.model import CandlestickInterval, Candlestick


def transform_candelstick_object_into_list(l: List[Candlestick]) -> List[List]:
    candelstick_class_attrs = [
        "openTime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closeTime",
        "quoteAssetVolume",
        "numTrades",
        "takerBuyBaseAssetVolume",
        "takerBuyQuoteAssetVolume",
        "ignore",
    ]
    res = []
    for c in l:
        _t = []
        for attr in candelstick_class_attrs:
            _t.append(getattr(c, attr))
        res.append(_t)
    return res


def _get_earliest_valid_timestamp(get_klines, symbol, interval):
    """Get earliest valid open timestamp from Binance

    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Binance Kline interval
    :type interval: str

    :return: first valid timestamp

    """
    kline = get_klines(
        symbol=symbol, interval=interval, limit=1, startTime=0, endTime=None
    )
    kline = transform_candelstick_object_into_list(kline)
    return kline[0][0]


def get_usdtfutures_historical_klines(
    get_klines: Callable, symbol, interval, start_str, end_str=None, limit=500
):
    """Get Historical Klines from Binance

    See dateparser docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/

    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Binance Kline interval
    :type interval: str
    :param start_str: Start date string in UTC format or timestamp in milliseconds
    :type start_str: str|int
    :param end_str: optional - end date string in UTC format or timestamp in milliseconds (default will fetch everything up to now)
    :type end_str: str|int
    :param limit: Default 500; max 1000.
    :type limit: int

    :return: list of OHLCV values

    """
    # init our list
    output_data = []

    # setup the max limit
    limit = limit

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    # convert our date strings to milliseconds
    if type(start_str) == int:
        start_ts = start_str
    else:
        start_ts = date_to_milliseconds(start_str)

    # establish first available start timestamp
    first_valid_ts = _get_earliest_valid_timestamp(get_klines, symbol, interval)
    start_ts = max(start_ts, first_valid_ts)

    # if an end time was passed convert it
    end_ts = None
    if end_str:
        if type(end_str) == int:
            end_ts = end_str
        else:
            end_ts = date_to_milliseconds(end_str)

    idx = 0
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
        temp_data = get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts,
        )
        temp_data = transform_candelstick_object_into_list(temp_data)

        # handle the case where exactly the limit amount of data was returned last loop
        if not len(temp_data):
            break

        # append this loops data to our output data
        output_data += temp_data

        # set our start timestamp using the last value in the array
        start_ts = temp_data[-1][0]

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # increment next call by our timeframe
        start_ts += timeframe

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)

    return output_data
