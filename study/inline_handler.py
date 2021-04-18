import requests
import telegram
import json
from telegram import Update, ParseMode, Bot, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
from setup import TOKEN, PROXY
from study.dict_controller import *
from study.user import User

CALLBACK_BUTTON_HMM = "model_hmm_sphinx"
CALLBACK_BUTTON_NEURAL = "model_neural_azure"

CALLBACK_BUTTON_EACH = "test_each_phoneme"
CALLBACK_BUTTON_COMPLEX = "test_random_5"
CALLBACK_BUTTON_SURVEY = "test_3c_3v"

CALLBACK_BUTTON_CONSONANTS = "test_consonants"
CALLBACK_BUTTON_VOWELS = "test_vowels"
CALLBACK_BUTTON_FINISH_TEST = "test_finish"

bot = Bot(
    token=TOKEN,
    # base_url=PROXY,  # delete it if connection via VPN
)


class InlineKeyboardFactory:
    @staticmethod
    def get_learn_mode_keyboard() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("Each phoneme", callback_data=CALLBACK_BUTTON_EACH)
            ],
            [
                InlineKeyboardButton("Complex", callback_data=CALLBACK_BUTTON_COMPLEX)
            ],
            [
                InlineKeyboardButton("Survey", callback_data=CALLBACK_BUTTON_SURVEY)
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

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

    @staticmethod
    def get_finish_test_keyboard() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("Stop checking and view results", callback_data=CALLBACK_BUTTON_FINISH_TEST)
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def get_model_type_keyboard() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("Hidden Markov Model", callback_data=CALLBACK_BUTTON_HMM)
            ],
            [
                InlineKeyboardButton("Neural Network (Azure Speech)", callback_data=CALLBACK_BUTTON_NEURAL)
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

        if data == CALLBACK_BUTTON_HMM:
            user = User(chat_id, is_testing=False, phone_dict=None, model_type=0)
            user.save_data()
            update.effective_message.reply_text(text='Please, select learning mode:',
                                                reply_markup=InlineKeyboardFactory.get_learn_mode_keyboard())

        elif data == CALLBACK_BUTTON_NEURAL:
            user = User(chat_id, is_testing=False, phone_dict=None, model_type=1)
            user.save_data()
            update.effective_message.reply_text(text='Please, select learning mode:',
                                                reply_markup=InlineKeyboardFactory.get_learn_mode_keyboard())

        elif data == CALLBACK_BUTTON_COMPLEX:
            user = proc.unpack_user_data(chat_id)
            user.mode = CALLBACK_BUTTON_COMPLEX
            sample = PhoneDict()  # used only for accessing JSON file to get 5 random phonemes next line
            rnd_5 = PhoneDict(custom_dict=sample.get_n_random(5), iter_type=0, one_example=True)
            user.phone_dict = rnd_5
            user.is_testing = True
            user.save_data()
            proc.display_question(update)
            # RANDOM 5 WORDS DICT

        elif data == CALLBACK_BUTTON_SURVEY:
            user = proc.unpack_user_data(chat_id)
            user.mode = CALLBACK_BUTTON_COMPLEX
            sample = PhoneDict()  # used only for accessing JSON file
            rnd_3c_3v = PhoneDict(custom_dict=sample.get_n_random(n_consonants=3, n_vowels=3), one_example=True)
            user.phone_dict = rnd_3c_3v
            user.is_testing = True
            user.save_data()
            proc.display_question(update)
            # RANDOM 3 CONSONANTS, 3 VOWELS, MISTAKES => REPEAT

        elif data == CALLBACK_BUTTON_EACH:
            user = proc.unpack_user_data(chat_id)
            user.mode = CALLBACK_BUTTON_EACH
            user.save_data()
            update.effective_message.reply_text(text='Please, select phonemes:',
                                                reply_markup=InlineKeyboardFactory.get_phone_type_keyboard())

        elif data == CALLBACK_BUTTON_CONSONANTS:
            consonants = PhoneDict(phone_type=0, auto_next=True, iter_type=0)
            user = proc.unpack_user_data(chat_id)
            user.is_testing = True
            user.phone_dict = consonants
            user.save_data()
            proc.display_question(update)

        elif data == CALLBACK_BUTTON_VOWELS:
            vowels = PhoneDict(phone_type=1, auto_next=True, iter_type=0)
            user = proc.unpack_user_data(chat_id)
            user.is_testing = True
            user.phone_dict = vowels
            user.save_data()
            proc.display_question(update)

        elif data == CALLBACK_BUTTON_FINISH_TEST:
            user = proc.unpack_user_data(chat_id)
            user.is_testing = False
            user.save_data()
            #proc.summary_test(chat_id)
