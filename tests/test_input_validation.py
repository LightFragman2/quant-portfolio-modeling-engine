import pytest

from datetime import date

from src.input_validation import (
    normalize_symbol,
    parse_symbol_input,
    validate_date_range,
)


def test_normalize_symbol():
    assert normalize_symbol(
        " aapl "
    ) == "AAPL"

    assert normalize_symbol(
        "$msft"
    ) == "MSFT"


def test_parse_symbol_input():
    result = parse_symbol_input(
        "aapl, MSFT nvda; JPM, AAPL"
    )

    assert result == [
        "AAPL",
        "MSFT",
        "NVDA",
        "JPM",
    ]


def test_invalid_symbol():
    with pytest.raises(
        ValueError
    ):
        normalize_symbol(
            "BAD SYMBOL!"
        )


def test_too_many_symbols():
    symbols = ",".join(
        f"S{i}"
        for i in range(11)
    )

    with pytest.raises(
        ValueError
    ):
        parse_symbol_input(
            symbols,
            max_symbols=10,
        )


def test_date_range():
    validate_date_range(
        date(2021, 1, 1),
        date(2022, 1, 1),
    )

    with pytest.raises(
        ValueError
    ):
        validate_date_range(
            date(2022, 1, 1),
            date(2021, 1, 1),
        )