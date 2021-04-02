import json


class PhoneDict:
    def __init__(self, path=None, iter_type=False):
        self.current = -1
        self.path = path if path else './sources/phonemes.json'
        self.iter_type = iter_type  # 0 - consonants, 1 - vowels
        self.data = self.__load_dict()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_type:
            self.current += 1
            if self.current < len(self.data['vowels']):
                element = tuple(self.data['vowels'].items())[self.current]
                return {'phone': element[0], 'examples': element[1]['examples']}
            else:
                raise StopIteration
        else:
            self.current += 1
            if self.current < len(self.data['consonants']):
                element = tuple(self.data['consonants'].items())[self.current]
                return {'phone': element[0], 'examples': element[1]['examples']}
            else:
                raise StopIteration

    def __load_dict(self):
        with open(self.path, 'r') as handle:
            return json.load(handle)

    def get_consonants(self):
        return self.data['consonants']

    def get_vowels(self):
        return self.data['vowels']


