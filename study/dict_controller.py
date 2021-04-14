import os
import json
import pickle
import random


class PhoneDict:
    def __init__(self, path=None, phone_type=False, iter_type=False, auto_next=True, custom_dict=None,
                 one_example: bool = False):

        # Current testing data
        self.current_phone_id = -1
        self.current_phone = ''
        self.current_examples = []
        self.current_example_word = ''
        self.current_help = ''
        self.current_attempt = 0

        # Path to source dict json (consonants and vowels)
        self.path = path if path else './sources/phonemes.json'

        # Dict parameters
        self.phone_type = phone_type  # 0 - consonants, 1 - vowels
        self.iter_type = iter_type  # 0 - examples, 1 - phones
        self.auto_next = auto_next
        self.one_example = one_example

        # Get data not to access file multiple times
        # Custom dict could be used instead of standard consonants & vowels system
        self.data = custom_dict if custom_dict else self.__load_dict()
        self.is_custom = custom_dict is not None

    def __iter__(self):
        return self

    def __next__(self, is_next_phone=False):
        # If cycle on phonemes or manually switched to the next phoneme
        if self.iter_type or is_next_phone:
            self.current_phone_id += 1
            element = None
            if self.is_custom:
                if self.current_phone_id < len(self.data):
                    element = tuple(self.data.items())[self.current_phone_id]
                else:
                    raise StopIteration
            else:
                if self.phone_type:
                    if self.current_phone_id < len(self.data['vowels']):
                        element = tuple(self.data['vowels'].items())[self.current_phone_id]
                    else:
                        raise StopIteration
                else:
                    if self.current_phone_id < len(self.data['consonants']):
                        element = tuple(self.data['consonants'].items())[self.current_phone_id]
                    else:
                        raise StopIteration
            self.current_examples = [random.choice(element[1]['examples'])] if self.one_example \
                else element[1]['examples']
            self.current_phone = element[0]
            self.current_help = element[1]['help']
            return {'phone': self.current_phone, 'examples': self.current_examples, 'help': self.current_help}

        # If switch to next example of the current phoneme
        else:
            if self.current_phone_id == -1:
                # Get data on the first run
                self.__next__(is_next_phone=True)
            try:
                # Get & drop first example from list
                self.current_example_word = self.current_examples.pop(0)
                return {'phone': self.current_phone, 'example': self.current_example_word,
                        'help': self.current_help}
            except IndexError:
                if self.auto_next:
                    # If auto switch, get the next phone and pop example
                    self.__next__(is_next_phone=True)
                    self.current_example_word = self.current_examples.pop(0)
                    return {'phone': self.current_phone, 'example': self.current_example_word,
                            'help': self.current_help}
                else:
                    raise StopIteration

    def get_current(self):
        if self.is_custom:
            element = tuple(self.data.items())[self.current_phone_id]
        else:
            if self.iter_type:
                element = tuple(self.data['vowels'].items())[self.current_phone_id]
            else:
                element = tuple(self.data['consonants'].items())[self.current_phone_id]
        return {'phone': element[0], 'examples': element[1]['examples'], 'help': element[1]['help']}

    def save_current_dict(self, path):
        with open(os.path.join(path, 'phone_dict.pkl'), 'wb+') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    def __load_dict(self):
        with open(self.path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def get_consonants(self) -> list:
        return [c[0] for c in tuple(self.data['consonants'].items())]

    def get_vowels(self) -> list:
        return [v[0] for v in tuple(self.data['vowels'].items())]

    def get_n_random(self, n: int = None, n_consonants: int = None, n_vowels: int = None):
        if n:
            all_phones = self.get_consonants() + self.get_vowels()
            rnd_n = random.sample(all_phones, n)
            return {rnd: self.data['consonants'].get(rnd, self.data['vowels'].get(rnd)) for rnd in rnd_n}
        rnd_n = []
        if n_consonants:
            rnd_n += random.sample(self.get_consonants(), n_consonants)
        if n_vowels:
            rnd_n += random.sample(self.get_vowels(), n_vowels)
        return {rnd: self.data['consonants'].get(rnd, self.data['vowels'].get(rnd)) for rnd in rnd_n}
