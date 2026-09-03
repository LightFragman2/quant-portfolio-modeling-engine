import re


SYMBOL_PATTERN = re.compile(
    r"^[A-Z0-9][A-Z0-9.\-]{0,14}$"
)


def normalize_symbol(symbol):
    cleaned_symbol = (
        symbol
        .strip()
        .upper()
        .lstrip("$")
    )

    if not cleaned_symbol:
        raise ValueError(
            "Ticker symbol cannot be empty."
        )

    if not SYMBOL_PATTERN.fullmatch(
        cleaned_symbol
    ):
        raise ValueError(
            f"Invalid ticker symbol: "
            f"{cleaned_symbol}"
        )

    return cleaned_symbol


def parse_symbol_input(
    raw_symbols,
    max_symbols=10,
):
    if not raw_symbols.strip():
        raise ValueError(
            "Enter at least one ticker symbol."
        )

    cleaned_input = (
        raw_symbols
        .replace(";", ",")
        .replace("\n", ",")
    )

    raw_tokens = []

    for section in cleaned_input.split(","):
        raw_tokens.extend(
            section.split()
        )

    symbols = []

    for token in raw_tokens:
        symbol = normalize_symbol(
            token
        )

        if symbol not in symbols:
            symbols.append(
                symbol
            )

    if len(symbols) == 0:
        raise ValueError(
            "Enter at least one ticker symbol."
        )

    if len(symbols) > max_symbols:
        raise ValueError(
            f"Enter no more than "
            f"{max_symbols} assets."
        )

    return symbols


def validate_date_range(
    start_date,
    end_date,
):
    if start_date >= end_date:
        raise ValueError(
            "Start date must be before "
            "the end date."
        )