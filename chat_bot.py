#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import logging
import tg_commands as tg
import processing.functions as proc

from setup import PROXY, TOKEN
from telegram import Bot, Update
from study.inline_handler import InlineCallback
from telegram.ext import CallbackContext, CommandHandler, Filters, MessageHandler, Updater, CallbackQueryHandler

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.


def start(update: Update, context: CallbackContext):
    """Send a message when the command /start is issued."""
    update.message.reply_text(f'Привет, {update.effective_user.first_name}!')


def chat_help(update: Update, context: CallbackContext):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Введи команду /start для начала. ')


def echo(update: Update, context: CallbackContext):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def error(update: Update, context: CallbackContext):
    """Log Errors caused by Updates."""
    logger.warning(f'Update {update} caused error {context.error}')


def main():
    # bot = Bot(
    #     token=TOKEN,
    #     base_url=PROXY,  # delete it if connection via VPN
    # )
    # updater = Updater(bot=bot, use_context=True)

    # Connect via socks proxy
    REQUEST_KWARGS = {
        'proxy_url': PROXY,
        # Optional, if you need authentication:
        # 'urllib3_proxy_kwargs': {
        #     'username': 'name',
        #     'password': 'passwd',
        # }
    }

    updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)

    # on different commands - answer in Telegram
    updater.dispatcher.add_handler(CommandHandler('start', tg.command_start))
    updater.dispatcher.add_handler(CommandHandler('help', chat_help))

    # on noncommand i.e message - echo the message on Telegram
    updater.dispatcher.add_handler(MessageHandler(Filters.text, echo))

    # handle voice messages
    updater.dispatcher.add_handler(MessageHandler(Filters.voice, proc.handle_voice))

    # inline handler
    updater.dispatcher.add_handler(CallbackQueryHandler(callback=InlineCallback.handle_keyboard_callback))

    # log all errors
    updater.dispatcher.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    logger.info('Start Bot')
    main()
