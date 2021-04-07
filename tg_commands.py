import json
import csv
import os

import librosa
import soundfile as sf
import myprosody as mysp
import processing.functions as proc

from study.inline_handler import InlineKeyboardFactory as keyboard
from pathlib import Path
from datetime import datetime, timezone

from telegram import Update, ParseMode, Bot, ChatAction, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from telegram.ext import CallbackContext
from setup import TOKEN, PROXY
from telegram.ext import Filters
from telegram.error import TimedOut

bot = Bot(
    token=TOKEN,
    base_url=PROXY,  # delete it if connection via VPN
)


def command_start(update: Update, context: CallbackContext):
    """Send a message when the command /start is issued."""
    update.message.reply_text(f'Hi, {update.effective_user.first_name}!')
    update.message.reply_text('Please, type <b>/help</b> to see the list of commands.',
                              parse_mode=ParseMode.HTML)
    update.message.reply_text('Choose the processing model type:', reply_markup=keyboard.get_model_type_keyboard())
    return update.effective_user.first_name


def voice_to_phonemes(update: Update, context: CallbackContext) -> list:  # ЧАСТОТА: 16/32 кГц, МОНО
    """ This function prints phonemes of voice speech

    Function gets voice file by message id, temporarily puts voice .ogg file to directory
    user_id/voices/message_id.ogg, converts it to .wav format (with sample rate of 16 kHz,
    mono) and puts it to myprosody/datatset/audioFiles/message_id.wav, where it is produced
    into list of phonemes by Sphinx.

    :param update: Telegram Update class instance
    :param context: Telegram CallbackContext (unused)
    :return: Sends message to the chat and returns list of phonemes, eg. ['W', 'UW', 'P', 'S']
    :rtype: list
    """
    if (datetime.now(timezone.utc) - update.effective_message.date).days > 3:
        return []
    chat_id = update.message.chat.id
    file_path = f"{chat_id}\\voices\\{update.message.message_id}.ogg"
    wav_path = f'F:\\LangBot\\myprosody\\dataset\\audioFiles\\{update.message.message_id}.wav'

    update.message.voice.get_file().download(custom_path=file_path)

    data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    sf.write(wav_path, data, sample_rate)

    phonemes = proc.get_phonemes(wav_path)
    update.effective_message.reply_text(f"speech: " + ' '.join(phonemes))
    proc.get_words(wav_path)
    update.effective_message.reply_text(proc.levenshtein_distance(phonemes, 'speech'))
    os.remove(file_path)
    return phonemes


def begin_test(update: Update, context: CallbackContext) -> None:
    update.effective_message.reply_text(f"")
