"""
Telegram notifications for trade events.
"""
import logging
import requests

TELEGRAM_BOT_TOKEN = "8665665252:AAGPmfrH0bQw5PaN-xa7W2PtP8QM-ARpu2s"
TELEGRAM_CHAT_ID = "6681140758"

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> None:
    """Send a message to Baptiste via Telegram bot."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        if not resp.ok:
            logger.warning("Telegram notification failed: %s", resp.text)
    except Exception as exc:
        logger.warning("Telegram notification error: %s", exc)


def notify_trade_entry(symbol: str, side: str, price: float, quantity: float,
                       stop_loss: float, take_profit: float, regime: str, dryrun: bool) -> None:
    if dryrun:
        return
    emoji = "🟢" if side == "short" else "🔵"
    msg = (
        f"{emoji} <b>Trade ouvert — {symbol}</b>\n"
        f"Side : <b>{side.upper()}</b>\n"
        f"Prix : <b>${price:,.2f}</b>\n"
        f"Quantité : {quantity}\n"
        f"Stop Loss : ${stop_loss:,.2f}\n"
        f"Take Profit : ${take_profit:,.2f}\n"
        f"Régime : {regime}"
    )
    send_telegram(msg)


def notify_trade_exit(symbol: str, side: str, price: float, reason: str, dryrun: bool) -> None:
    if dryrun:
        return
    emoji = "🔴"
    msg = (
        f"{emoji} <b>Trade fermé — {symbol}</b>\n"
        f"Side : <b>{side.upper()}</b>\n"
        f"Prix de sortie : <b>${price:,.2f}</b>\n"
        f"Raison : {reason}"
    )
    send_telegram(msg)
