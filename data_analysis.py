#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     data_analysis.py
#
# Description:
# Version:      1.0
# Created:      2020/6/9 16:50
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#


import json
import matplotlib.pyplot as plt


def analysis(input_file):
    with open(input_file, 'r', encoding="utf-8") as fp:
        examples = []
        for line in fp:
            examples.append(json.loads(line))
        total_questions = []
        total_evidences = []
        total_options = []
        for ex in examples:
            total_questions.append(ex["question"])
            if len(ex["question"]) > 176:
                print(ex)
            if "evidences" in ex:
                for ev in ex["evidences"]:
                    total_evidences.append(ev)
            elif "analysis" in ex:
                total_evidences.append(ex["analysis"])
            for k, v in ex["options"].items():
                total_options.append(v)
                if len(v) > 64:
                    print(ex)

        fig = plt.figure()
        ax1 = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)

        def get_length_freq(texts):
            lens = [len(x) for x in texts]
            freq = {}
            for i in lens:
                if i not in freq:
                    freq[i] = 1
                else:
                    freq[i] += 1
            ls, fs = zip(*(freq.items()))
            return ls, fs
        ax1.scatter(*get_length_freq(total_questions))
        ax1.set_title("question")
        ax1.set_xlabel("length")
        ax1.set_ylabel("freq")
        ax2.scatter(*get_length_freq(total_options))
        ax2.set_title("options")
        ax2.set_xlabel("length")
        ax2.set_ylabel("freq")
        ax3.scatter(*get_length_freq(total_evidences))
        ax3.set_title("evidences")
        ax3.set_xlabel("length")
        ax3.set_ylabel("freq")
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    analysis("data/train.json")


