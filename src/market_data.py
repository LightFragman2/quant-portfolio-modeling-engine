import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from alpaca.data.enums import (
    Adjustment,
    DataFeed,
)
from alpaca.data.historical import (
    StockHistoricalDataClient,
)
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame


def get_alpaca_client():
    load_dotenv()

    api_key = os.getenv(
        "ALPACA_API_KEY"
    )

    secret_key = os.getenv(
        "ALPACA_SECRET_KEY"
    )

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials were not found. "
            "Add ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "to your .env file."
        )

    return StockHistoricalDataClient(
        api_key,
        secret_key,
    )


def get_historical_close_prices(
    symbols,
    start_date,
    end_date=None,
):
    if len(symbols) == 0:
        raise ValueError(
            "At least one symbol is required."
        )

    client = get_alpaca_client()

    if end_date is None:
        end_date = datetime.now(
            timezone.utc
        )

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(
        request
    )

    dataframe = (
        bars.df
        .reset_index()
    )

    if dataframe.empty:
        raise ValueError(
            "Alpaca returned no historical data."
        )

    price_table = dataframe.pivot(
        index="timestamp",
        columns="symbol",
        values="close",
    )

    price_table = (
        price_table
        .sort_index()
    )

    missing_symbols = [
        symbol
        for symbol in symbols
        if symbol not in price_table.columns
    ]

    if missing_symbols:
        raise ValueError(
            "No historical data was returned for: "
            + ", ".join(missing_symbols)
        )

    # Keep only dates where every asset has data.
    #
    # This is important because covariance and correlation
    # require observations from matching dates.
    price_table = price_table.dropna(
        subset=symbols
    )

    if len(price_table) < 2:
        raise ValueError(
            "Not enough aligned historical observations."
        )

    historical_prices = {}

    for symbol in symbols:
        historical_prices[symbol] = (
            price_table[symbol]
            .astype(float)
            .tolist()
        )

    dates = list(
        price_table.index
    )

    return historical_prices, dates


def get_latest_trade_prices(
    symbols,
):
    if len(symbols) == 0:
        raise ValueError(
            "At least one symbol is required."
        )

    client = get_alpaca_client()

    request = StockLatestTradeRequest(
        symbol_or_symbols=symbols,
        feed=DataFeed.IEX,
    )

    latest_trades = (
        client.get_stock_latest_trade(
            request
        )
    )

    prices = {}

    for symbol in symbols:
        if symbol not in latest_trades:
            raise ValueError(
                f"No latest trade returned for {symbol}."
            )

        trade = latest_trades[
            symbol
        ]

        prices[symbol] = {
            "price": float(
                trade.price
            ),
            "timestamp": trade.timestamp,
        }

    return prices