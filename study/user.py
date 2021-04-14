import os
import json
import pickle


class User:
    def __init__(self, chat_id, mode=None, is_testing=False, phone_dict=None, model_type=False):
        self.chat_id = chat_id
        self.is_testing = is_testing
        self.mode = mode    # all phones / complex / survey
        self.phone_dict = phone_dict
        self.model_type = model_type  # 0 is HMM, 1 is Neural network

    def save_data(self):
        with open(f"./{self.chat_id}/personal/user.pkl", 'wb+') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    def analyze_hmm(self):
        result_json_path = f"./{self.chat_id}/personal/hmm.json"
        with open(result_json_path, 'r') as handle:
            user_data = json.load(handle)
            if self.phone_dict:
                pass

    def analyze_neural(self):
        result_json_path = f"./{self.chat_id}/personal/neural.json"
        with open(result_json_path, 'r') as handle:
            pass
