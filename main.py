"""
Bear Trader — HyperLiquid Futures Algo Trading System
=====================================================

Usage examples:
  python main.py --mode backtest --strategy trend --symbol BTC/USDC:USDC
  python main.py --mode live --strategy trend --symbol BTC/USDC:USDC

Safety:
  HL_TESTNET=true in .env by default.
  Set HL_TESTNET=false only when ready to trade with real money.
"""

import argparse
import sys

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bear Trader — HyperLiquid Futures algorithmic trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "dryrun"],
        required=True,
        help="Operating mode",
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=["trend", "bull"],
        default=["trend"],
        help="Strategy or strategies to run (default: both)",
    )
    parser.add_argument(
        "--symbol",
        nargs="+",
        default=["BTC/USDC:USDC"],
        help="Trading symbol(s), e.g. BTC/USDC:USDC ETH/USDC:USDC",
    )
    parser.add_argument(
        "--start",
        default=config.BACKTEST_START,
        help=f"Backtest start date (default: {config.BACKTEST_START})",
    )
    parser.add_argument(
        "--end",
        default=config.BACKTEST_END,
        help=f"Backtest end date (default: {config.BACKTEST_END})",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=config.BACKTEST_INITIAL_CAPITAL,
        help=f"Initial capital in USDT (default: {config.BACKTEST_INITIAL_CAPITAL})",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetch all data from API (ignore cache)",
    )
    return parser.parse_args()


def run_backtest(args: argparse.Namespace) -> None:
    from data.cache import load_ohlcv
    from backtest.engine import run_backtest as engine_backtest
    from backtest.metrics import print_metrics
    from backtest.plotter import plot_equity_and_drawdown

    import signals.trend_following as tf_strategy
    import signals.trend_bull as bull_strategy

    for symbol in args.symbol:
        for strategy_name in args.strategy:
            logger.info(
                "Running backtest: strategy=%s symbol=%s from=%s to=%s capital=%.0f",
                strategy_name, symbol, args.start, args.end, args.capital,
            )

            # Generate signals
            if strategy_name == "trend":
                df = tf_strategy.generate_signals(
                    symbol, args.start, args.end, testnet=config.HL_TESTNET
                )
            elif strategy_name == "bull":
                df = bull_strategy.generate_signals(
                    symbol, args.start, args.end, testnet=config.HL_TESTNET
                )

            df.attrs["strategy"] = strategy_name

            # Run backtest
            result = engine_backtest(
                signal_df=df,
                initial_capital=args.capital,
            )

            # Print metrics
            print(f"\n{'='*55}")
            print(f"  Strategy: {strategy_name.upper()}  |  Symbol: {symbol}")
            print_metrics(result.metrics)

            # Save equity curve
            equity_path = f"backtest/results/{strategy_name}_{symbol.replace('/', '_').replace(':', '_')}_equity.csv"
            import os; os.makedirs("backtest/results", exist_ok=True)
            result.equity_curve.to_csv(equity_path)
            logger.info("Equity curve saved to %s", equity_path)

            # Plot
            plot_equity_and_drawdown(
                result.equity_curve,
                result.trades,
                symbol=symbol,
                strategy=strategy_name,
                save=True,
                show=False,
            )


def run_live_or_dryrun(args: argparse.Namespace) -> None:
    from live.bot import TradingBot

    if args.mode == "live" and not config.HL_TESTNET:
        print("\n⚠️  WARNING: You are about to trade with REAL MONEY on HyperLiquid!")
        print("   HL_TESTNET=false in your .env file.")
        confirm = input("   Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            print("Aborting.")
            sys.exit(0)

    bot = TradingBot(
        symbols=args.symbol,
        strategies=args.strategy,
        mode=args.mode,
        testnet=config.HL_TESTNET,
    )
    bot.run()


def main() -> None:
    args = parse_args()

    logger.info(
        "Bear Trader starting | mode=%s testnet=%s",
        args.mode, config.HL_TESTNET,
    )

    if args.mode == "backtest":
        run_backtest(args)
    else:
        run_live_or_dryrun(args)


if __name__ == "__main__":
    main()
