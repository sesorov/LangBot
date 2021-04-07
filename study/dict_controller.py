import os
import json
import pickle


class PhoneDict:
    def __init__(self, path=None, phone_type=False, iter_type=False, auto_next=True):

        # Current testing data
        self.current_phone_id = -1
        self.current_phone = ''
        self.current_examples = []
        self.current_example_word = ''

        # Path to source dict json (consonants and vowels)
        self.path = path if path else './sources/phonemes.json'

        # Dict parameters
        self.phone_type = phone_type  # 0 - consonants, 1 - vowels
        self.iter_type = iter_type  # 0 - examples, 1 - phones
        self.auto_next = auto_next

        # Get data not to access file multiple times
        self.data = self.__load_dict()

    def __iter__(self):
        return self

    def __next__(self, is_next_phone=False):

        # If cycle on phonemes or manually switched to the next phoneme
        if self.iter_type or is_next_phone:
            self.current_phone_id += 1
            if self.phone_type:
                if self.current_phone_id < len(self.data['vowels']):
                    element = tuple(self.data['vowels'].items())[self.current_phone_id]
                    self.current_examples = element[1]['examples']
                    self.current_phone = element[0]
                    return {'phone': element[0], 'examples': element[1]['examples']}
                else:
                    self.is_testing = False
                    raise StopIteration
            else:
                if self.current_phone_id < len(self.data['consonants']):
                    element = tuple(self.data['consonants'].items())[self.current_phone_id]
                    self.current_examples = element[1]['examples']
                    self.current_phone = element[0]
                    return {'phone': element[0], 'examples': element[1]['examples']}
                else:
                    self.is_testing = False
                    raise StopIteration

        # If switch to next example of the current phoneme
        else:
            if self.current_phone_id == -1:
                # Get data on the first run
                self.__next__(is_next_phone=True)
            try:
                # Get & drop first example from list
                self.current_example_word = self.current_examples.pop(0)
                return {'phone': self.current_phone, 'example': self.current_example_word}
            except IndexError:
                if self.auto_next:
                    # If auto switch, get the next phone and pop example
                    self.__next__(is_next_phone=True)
                    self.current_example_word = self.current_examples.pop(0)
                    return {'phone': self.current_phone, 'example': self.current_example_word}
                else:
                    raise StopIteration

    def get_current(self):
        if self.iter_type:
            element = tuple(self.data['vowels'].items())[self.current_phone_id]
            return {'phone': element[0], 'examples': element[1]['examples']}
        else:
            element = tuple(self.data['consonants'].items())[self.current_phone_id]
            return {'phone': element[0], 'examples': element[1]['examples']}

    def save_current_dict(self, path):
        with open(os.path.join(path, 'phone_dict.pkl'), 'wb+') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    def __load_dict(self):
        with open(self.path, 'r') as handle:
            return json.load(handle)

    def get_consonants(self):
        return self.data['consonants']

    def get_vowels(self):
        return self.data['vowels']
