import os
import librosa
import pyaudio
import soundfile as sf
import myprosody as mysp

from telegram import Update
from os import path
from pocketsphinx import AudioFile, get_model_path, get_data_path
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

    #p = pyaudio.PyAudio()
    #stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    #stream.start_stream()
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
