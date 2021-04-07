import os
import pickle


class User:
    def __init__(self, chat_id, is_testing=False, phone_dict=None, model_type=False):
        self.chat_id = chat_id
        self.is_testing = is_testing
        self.phone_dict = phone_dict
        self.model_type = model_type  # 0 is HMM, 1 is Neural network

    def save_data(self):
        with open(os.path.join(f"F:\\LangBot\\{self.chat_id}\\personal", 'user.pkl'), 'wb+') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)
