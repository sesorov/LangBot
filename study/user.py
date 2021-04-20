import os
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from statistics import mean
from functools import reduce


class User:
    def __init__(self, chat_id, mode=None, is_testing=False, phone_dict=None, model_type=False):
        self.chat_id = chat_id
        self.is_testing = is_testing
        self.mode = mode  # all phones / complex / survey
        self.phone_dict = phone_dict
        self.model_type = model_type  # 0 is HMM, 1 is Neural network

    def save_data(self):
        with open(f"./{self.chat_id}/personal/user.pkl", 'wb+') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    def analyze_hmm(self, output_type=0):  # 0 - text, 1 - graph
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
            user_data = json.load(handle)
            if self.phone_dict.current_attempt:
                self.is_testing = False
                self.save_data()

                if output_type:
                    # Results on first attempt
                    pre = [value for value in user_data if value['attempt'] == 0]
                    # Results on second attempt
                    post = [value for value in user_data if value['attempt'] == 1]

                    # Building bar chart for overall overview of pronunciation score and accuracy
                    df_pre = pd.DataFrame(pre)
                    df_post = pd.DataFrame(post)

                    df = pd.merge(df_pre, df_post, on='phone')
                    df.plot.bar(x='phone', y=['score_x', 'score_y', 'accuracy_x', 'accuracy_y'],
                                label=['Score: 1st', 'Score: 2nd', 'Accuracy: 1st', 'Accuracy: 2nd'],
                                title='Higher is better', width=.75)
                    plt.xticks(rotation=0)
                    plt.legend(loc='lower right')
                    plt.savefig(f"./{self.chat_id}/personal/overall_stats_neural.jpg")

                    # Building target phoneme score graphs (individual analysis for each phoneme)
                    pre_phone = [{'phone': entry['phone'], 'score': entry['phonemes_scores'][entry['phone']]}
                                 for entry in pre]
                    post_phone = [{'phone': entry['phone'], 'score': entry['phonemes_scores'][entry['phone']]}
                                  for entry in post]
                    df_pre_phone = pd.DataFrame(pre_phone)
                    df_post_phone = pd.DataFrame(post_phone)

                    ax = df_pre_phone.plot(x='phone', y='score', label='1st attempt')
                    df_post_phone.plot(x='phone', y='score', ax=ax, label='2nd attempt', title='Higher is better')
                    plt.savefig(f"./{self.chat_id}/personal/target_phone_stats.jpg")

                    return f"./{self.chat_id}/personal/stats.jpg"
                else:
                    pre_score = [value['score'] for value in user_data if value['attempt'] == 0]
                    pre_accuracy = [value['accuracy'] for value in user_data if value['attempt'] == 0]

                    post_score = [value['score'] for value in user_data if value['attempt'] == 1]
                    post_accuracy = [value['accuracy'] for value in user_data if value['attempt'] == 1]

                    pre_avg_phone = round(mean([value['phonemes_scores'][value['phone']] for value in user_data if
                                                value['attempt'] == 0]), 1)
                    post_avg_phone = round(mean([value['phonemes_scores'][value['phone']] for value in user_data if
                                                 value['attempt'] == 1]), 1)

                    return f"""
Results: {self.mode}
Model: {'Neural' if self.model_type else 'Hidden Markov Model'}

First attempt:
Average word pronunciation score (higher is better): {round(mean(pre_score), 1)}%
Average accuracy score (higher is better): {round(mean(pre_accuracy), 1)}%
Average target phone pronunciation correctness: {pre_avg_phone}%

Second attempt:
Average word pronunciation score (higher is better): {round(mean(post_score), 1)}%
Average accuracy score (higher is better): {round(mean(post_accuracy), 1)}%
Average target phone pronunciation correctness: {post_avg_phone}%

You may improve your results by practising each phoneme individually in corresponding mode.
"""
            else:
                self.phone_dict.current_phone_id = -1
                self.phone_dict.current_attempt = 1
                self.save_data()
                return False
