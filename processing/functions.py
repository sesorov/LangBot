import os
import numpy as np
import librosa
import soundfile as sf

from os import path
from pocketsphinx.pocketsphinx import *
from telegram import Update, ParseMode, Bot, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
from telegram.ext import CallbackContext
from setup import TOKEN, PROXY
from study.dict_controller import *
from datetime import datetime, timezone
from pathlib import Path
from study.inline_handler import InlineKeyboardFactory as keyboard
from study.inline_handler import CALLBACK_BUTTON_EACH, CALLBACK_BUTTON_COMPLEX, CALLBACK_BUTTON_SURVEY
from azure_speech.analysis import Analysis

bot = Bot(
    token=TOKEN,
    base_url=PROXY,  # delete it if connection via VPN
)


def handle_voice(update: Update, context: CallbackContext):
    chat_id = update.message.chat.id

    # Getting current user state
    try:
        user = unpack_user_data(chat_id)
        if user.is_testing:
            if user.model_type:
                voice_analyze_neural(update, user.phone_dict)
            else:
                voice_analyze_hmm(update, user.phone_dict)
            display_question(update)
            return
    except FileNotFoundError:
        # User hasn't yet started the test
        pass
    update.effective_message.reply_text(text='You need to be in learning mode to check your pronunciation.\n' +
                                             'Please, choose the learning mode:',
                                        reply_markup=keyboard.get_learn_mode_keyboard())


def display_question(update: Update):
    chat_id = update.effective_message.chat_id
    user = unpack_user_data(chat_id)
    try:
        current = user.phone_dict.__next__()
        update.effective_message.reply_text(text=f"[{current['phone']}] Pronounce: {current['example']}")
        user.save_data()
    except StopIteration:
        user.save_data()
        update.effective_message.reply_text(text='This is the end of the test.')


def unpack_user_data(chat_id):
    import pickle

    with open(f"./{chat_id}/personal/user.pkl", 'rb') as load:
        return pickle.load(load)


def update_user_test_data(chat_id, data: dict, name: str):
    datafile = Path(f"./{chat_id}/personal/{name}.json")
    user_data = []
    try:
        with open(datafile, 'r') as handle:
            user_data = json.load(handle)
    except FileNotFoundError:
        pass
    user_data.append(data)
    with open(datafile, 'w+') as handle:
        json.dump(user_data, handle, indent=4)


def voice_analyze_neural(update: Update, phone_dict: PhoneDict):
    if (datetime.now(timezone.utc) - update.effective_message.date).days > 3:
        return []
    chat_id = update.message.chat.id
    file_path = f"{chat_id}\\voices\\{update.message.message_id}.ogg"
    wav_path = f'F:\\LangBot\\myprosody\\dataset\\audioFiles\\{update.message.message_id}.wav'

    update.message.voice.get_file().download(custom_path=file_path)

    data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    sf.write(wav_path, data, sample_rate)

    example_word = phone_dict.current_example_word
    phone = phone_dict.current_phone

    if example_word:
        analysis = Analysis(chat_id)
        result = analysis.analyse_and_save(wav_path, example_word)
        update_user_test_data(chat_id, data={'score': result['pronunciation_score'],
                                             'accuracy': result['accuracy_score'],
                                             'complereness': result['completeness_score'],
                                             'fluency': result['fluency_score'],
                                             'error_type': result['error_type'][0],
                                             'phonemes': result['phonemes'],
                                             'phone': phone,
                                             'attempt': phone_dict.current_attempt},
                              name='neural')
        update.effective_message.reply_text(f"Pronunciation score: {result['pronunciation_score']} (higher is better)" +
                                            f"\nAccuracy score: {result['accuracy_score']}" +
                                            f"\nCompleteness score: {result['completeness_score']}" +
                                            f"\nFluency score: {result['fluency_score']}" +
                                            f"\nError type: {result['error_type'][0]}" +
                                            f"\nPhonemes list: {' '.join(result['phonemes'])}")
    os.remove(file_path)


def voice_analyze_hmm(update: Update, phone_dict: PhoneDict):
    example_word = phone_dict.current_example_word
    target_phone = phone_dict.current_phone
    if (datetime.now(timezone.utc) - update.effective_message.date).days > 3:
        return []
    chat_id = update.message.chat.id
    file_path = f"{chat_id}\\voices\\{update.message.message_id}.ogg"
    wav_path = f'F:\\LangBot\\myprosody\\dataset\\audioFiles\\{update.message.message_id}.wav'

    update.message.voice.get_file().download(custom_path=file_path)

    data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    sf.write(wav_path, data, sample_rate)

    phonemes = get_phonemes(wav_path)
    update.effective_message.reply_text(f"Speech: " + ' '.join(phonemes))

    score = levenshtein_distance_sphinx(phonemes, example_word)
    is_correct = is_correct_phone(phonemes, target_phone)
    update.effective_message.reply_text(f"Rate: {score} (lower is better)")
    if not is_correct:
        update.effective_message.reply_text('Mispronunciation detected. Here are some tips:')
        update.effective_message.reply_text(phone_dict.get_current()['help'])
    update_user_test_data(chat_id, data={'phone': target_phone, 'example': example_word,
                                         'score': score, 'phoneme_correct': is_correct,
                                         'attempt': phone_dict.current_attempt}, name='hmm')
    os.remove(file_path)
    return phonemes


def is_correct_phone(test_phonemes, target_phoneme):
    for phone in test_phonemes:
        if levenshtein_distance(phone, target_phoneme) == 0:
            return True
    return False


def get_phonemes(voice_path: str) -> list:
    """ This function returns phonemes of speech

    Function gets voice_path, configures custom model for phoneme recognition,
    creates the decoder based on config and processes input audio by segments.

    :param voice_path: Path to voice .wav file (16 kHz, mono)
    :return: List of phonemes (SIL is "silence")
    :rtype: list
    """

    model_path = "custom_model_en_us-v52/en-us"
    config = Decoder.default_config()
    config.set_string('-hmm', path.join(model_path, 'en-us'))
    config.set_string('-dict', path.join(model_path, 'cmudict-en-us.dict'))
    config.set_string('-allphone', path.join(model_path, 'en-us-phone.lm.bin'))  # dmp
    config.set_float('-lw', 2.0)
    config.set_float('-beam', 1e-10)
    config.set_float('-pbeam', 1e-10)

    # Decode streaming data.
    decoder = Decoder(config)

    decoder.start_utt()

    stream = open(voice_path, 'rb')
    while True:
        buf = stream.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
    decoder.end_utt()

    hypothesis = decoder.hyp()
    return [seg.word for seg in decoder.seg()]


def get_words(voice_path: str) -> None:
    """ This function returns words of speech

    Function gets voice_path, configures custom model for words recognition,
    creates the decoder based on config and processes input audio by segments.

    :param voice_path: Path to voice .wav file (16 kHz, mono)
    :return: Prints best hypothesis matches
    :rtype: None
    """
    model_path = "custom_model_en_us-v52/en-us"
    config = Decoder.default_config()
    config.set_string('-hmm', os.path.join(model_path, 'en-us'))
    config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
    config.set_string('-dict', os.path.join(model_path, 'cmudict-en-us.dict'))
    config.set_float('-kws_threshold', 1e+20)

    # p = pyaudio.PyAudio()
    # stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    # stream.start_stream()
    stream = open(voice_path, 'rb')

    # Process audio chunk by chunk. On keyword detected perform action and restart search
    decoder = Decoder(config)
    decoder.start_utt()

    while True:
        buf = stream.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
        if decoder.hyp() is not None:
            print(decoder.hyp().hypstr)
            print('Best hypothesis segments: ', [(seg.word, seg.prob) for seg in decoder.seg()])


def levenshtein_distance(test_sequence: list or str, target_sequence: list or str):
    test_sequence = list(test_sequence)
    target_sequence = list(target_sequence)

    try:
        test_sequence.remove('SIL')
        test_sequence.remove('+SPN+')
    except ValueError:
        pass

    size_x = len(test_sequence) + 1
    size_y = len(target_sequence) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if test_sequence[x - 1] == target_sequence[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # deletion
                    matrix[x - 1, y - 1],  # replacement
                    matrix[x, y - 1] + 1)  # addition
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # deletion
                    matrix[x - 1, y - 1] + 1,  # replacement
                    matrix[x, y - 1] + 1)  # addition
    return matrix[size_x - 1, size_y - 1]


def levenshtein_distance_sphinx(test_phonemes: list, word: str):
    # setting up the model
    model_path = "custom_model_en_us-v52/en-us"
    config = Decoder.default_config()
    config.set_string('-hmm', os.path.join(model_path, 'en-us'))
    config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
    config.set_string('-dict', os.path.join(model_path, 'cmudict-en-us.dict'))
    config.set_float('-kws_threshold', 1e+20)
    decoder = Decoder(config)
    decoder.start_utt()

    test_phonemes.remove('SIL')
    correct = decoder.lookup_word(word).split()

    return levenshtein_distance(test_phonemes, correct)
