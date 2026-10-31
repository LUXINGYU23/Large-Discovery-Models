"""Portfolio constraints shared by source preflight and the Assay worker."""

from dataclasses import fields


def portfolio_config(protocol, config_type):
    settings = protocol.assay_portfolio
    if set(settings) != {item.name for item in fields(config_type)}:
        raise ValueError("freeze the full independent PortfolioBacktestConfig in assay_portfolio")
    config = config_type(**settings)
    market = "A" if protocol.market.startswith("csi") else "US"
    if config.market != market or config.universe != protocol.market.upper():
        raise ValueError("Assay portfolio market/universe differs from factor evaluation")
    if config.benchmark != "custom" or not config.benchmark_symbol:
        raise ValueError("Assay portfolio requires an actual custom index benchmark")
    if config.execution_price not in {"next_open", "next_close"}:
        raise ValueError("only actual open/close execution is supported by the pinned portfolio engine")
    if config.warmup_days or not config.save_trade_log or not config.save_position_log or config.output_frequency != "daily":
        raise ValueError("portfolio must retain all daily NAV, trades and positions after factor warmup")
    if config.sector_neutral or config.include_bid_ask or config.northbound_flow_filter or config.sz_sh_connect_only or config.inclusion_anticipation:
        raise ValueError("portfolio requests controls without an implemented data input")
    if config.slippage_model != "zero" or config.new_listing_lockout_days or config.ipo_lockout_days or config.rebalance_around_index or config.st_filter:
        raise ValueError("portfolio requests impact or filters without an implemented data input")
    return config
