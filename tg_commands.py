import json
import csv
import os

import librosa
import soundfile as sf
import myprosody as mysp
import processing.functions as proc

from pathlib import Path
from datetime import datetime, timezone

from sphi_test import test_voice

from telegram import Update, ParseMode, Bot, ChatAction, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from telegram.ext import CallbackContext
from setup import TOKEN, PROXY
from telegram.ext import Filters
from telegram.error import TimedOut
from telegram import ChatAction, Update

bot = Bot(
    token=TOKEN,
    base_url=PROXY,  # delete it if connection via VPN
)


def command_start(update: Update, context: CallbackContext):
    """Send a message when the command /start is issued."""
    update.message.reply_text(f'Hi, {update.effective_user.first_name}!')
    update.message.reply_text('Please, type <b>/help</b> to see the list of commands.',
                              parse_mode=ParseMode.HTML)
    return update.effective_user.first_name


def voice_to_text(update: Update, context: CallbackContext) -> None:  # ЧАСТОТА: 16/32 кГц, МОНО
    """ This function prints phonemes of voice speech

    Function gets voice file by message id, temporarily puts voice .ogg file to directory
    user_id/voices/message_id.ogg, converts it to .wav format (with sample rate of 16 kHz,
    mono) and puts it to myprosody/datatset/audioFiles/message_id.wav, where it is produced
    into list of phonemes by Sphinx.

    :param update: Telegram Update class instance
    :param context: Telegram CallbackContext (unused)
    :return: List of phonemes (SIL is "silence")
    :rtype: list
    """
    if (datetime.now(timezone.utc) - update.effective_message.date).days > 3:
        return
    chat_id = update.message.chat.id
    file_path = f"{chat_id}\\voices\\{update.message.message_id}.ogg"
    wav_path = f'F:\\LangBot\\myprosody\\dataset\\audioFiles\\{update.message.message_id}.wav'

    update.message.voice.get_file().download(custom_path=file_path)

    data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    sf.write(wav_path, data, sample_rate)

    update.effective_message.reply_text(proc.get_phonemes(wav_path))
    proc.get_words(wav_path)
    p = f"{update.message.message_id}"
    c = r"F:\LangBot\myprosody"  # an example of path to directory "myprosody"
    update.effective_message.reply_text(mysp.mysppron(p, c))
    os.remove(file_path)
    os.remove(f'F:\\LangBot\\myprosody\\dataset\\audioFiles\\{update.message.message_id}.TextGrid')
    os.remove(f'F:\\LangBot\\myprosody\\dataset\\audioFiles\\{update.message.message_id}.wav')

    to_gs = update.message.voice.duration > 58
