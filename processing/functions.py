import os
import numpy as np

from os import path
from pocketsphinx.pocketsphinx import *


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


def levenshtein_distance(test_phonemes: list, word: str):
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

    size_x = len(test_phonemes) + 1
    size_y = len(correct) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if test_phonemes[x - 1] == correct[y - 1]:
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
