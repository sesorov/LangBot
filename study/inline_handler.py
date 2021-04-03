import requests
import telegram
import json
from telegram import Update, ParseMode, Bot, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
from setup import TOKEN, PROXY
from study.standard import *

CALLBACK_BUTTON_CONSONANTS = "test_consonants"
CALLBACK_BUTTON_VOWELS = "test_vowels"

bot = Bot(
    token=TOKEN,
    # base_url=PROXY,  # delete it if connection via VPN
)


class InlineKeyboardFactory:
    @staticmethod
    def get_phone_type_keyboard() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("Consonants", callback_data=CALLBACK_BUTTON_CONSONANTS)
            ],
            [
                InlineKeyboardButton("Vowels", callback_data=CALLBACK_BUTTON_VOWELS)
            ]
        ]
        return InlineKeyboardMarkup(keyboard)


class InlineCallback:
    @staticmethod
    def handle_keyboard_callback(update: Update, context=None):  # Gets callback_data from the pushed button
        query = update.callback_query  # Gets query from callback
        data = query.data  # callback_data of pushed button
        chat_id = update.effective_message.chat_id  # chat id for sending messages

        import processing.functions as proc

        if data == CALLBACK_BUTTON_CONSONANTS:
            consonants = PhoneDict(iter_type=0)
            proc.update_user_test_data(chat_id, data={'is_testing': True})
            proc.display_question(update, consonants, is_next=True)

        elif data == CALLBACK_BUTTON_VOWELS:
            vowels = PhoneDict(iter_type=1)
            proc.update_user_test_data(chat_id, data={'is_testing': True})
            proc.display_question(update, vowels, is_next=True)
