import os
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from statistics import mean


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

    def analyze_hmm(self, output_type=0):   # 0 - text, 1 - graph
        result_json_path = f"./{self.chat_id}/personal/hmm.json"
        with open(result_json_path, 'r') as handle:
            user_data = json.load(handle)
            if self.phone_dict.current_attempt:
                self.is_testing = False
                self.save_data()

                if output_type:
                    pre = [value for value in user_data if value['attempt'] == 0]
                    post = [value for value in user_data if value['attempt'] == 1]
                    df_pre = pd.DataFrame(pre)
                    df_post = pd.DataFrame(post)
                    ax = df_pre.plot(x='phone', y='score', label='First')
                    df_post.plot(x='phone', y='score', ax=ax, label='Second', title='Lower is better')
                    plt.savefig(f"./{self.chat_id}/personal/stats.jpg")
                    return f"./{self.chat_id}/personal/stats.jpg"
                else:
                    pre_score = [value['score'] for value in user_data if value['attempt'] == 0]
                    pre_correct_num = len([value for value in user_data if value['attempt'] == 0
                                           and value['phoneme_correct']])

                    post_score = [value['score'] for value in user_data if value['attempt'] == 1]
                    post_correct_num = len([value for value in user_data if value['attempt'] == 1
                                           and value['phoneme_correct']])

                    return f"""
Results: {self.mode}
Model: {'Neural' if self.model_type else 'Hidden Markov Model'}

First attempt:
Average word pronunciation score (lower is better): {round(mean(pre_score), 1)}
Num of correct phonemes: {pre_correct_num}/{len(pre_score)}

Second attempt:
Average word pronunciation score (lower is better): {round(mean(post_score), 1)}
Num of correct phonemes: {post_correct_num}/{len(post_score)}

You may improve your results by practising each phoneme individually in corresponding mode.
"""
            else:
                self.phone_dict.current_phone_id = -1
                self.phone_dict.current_attempt = 1
                self.save_data()
                return False

    def analyze_neural(self, output_type):
        result_json_path = f"./{self.chat_id}/personal/neural.json"
        with open(result_json_path, 'r') as handle:
            if self.phone_dict.current_attempt:
                self.is_testing = False
                self.save_data()

                if output_type:
                    pass
                else:
                    pass
                return True
            else:
                self.phone_dict.current_phone_id = -1
                self.phone_dict.current_attempt = 1
                self.save_data()
                return False
