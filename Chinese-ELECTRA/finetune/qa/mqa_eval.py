#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     mqa_eval.py
#
# Description:  evaluation for multi-choice mrc
# Version:      1.0
# Created:      2020/6/8 10:59
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
import re
import json
import tensorflow.compat.v1 as tf
from collections import Counter

import configure_finetuning


def read_predictions(prediction_file):
    with tf.io.gfile.GFile(prediction_file) as f:
        predictions = json.load(f)
    return predictions


def read_answers(gold_file):
    answers = {}
    samples = {}
    with tf.io.gfile.GFile(gold_file, 'r') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            answers[sample["key"]] = sample["answers"]
            answers[sample["key"]] = sample
    return answers, samples


def evaluate(answers, predictions, samples, bad_file):
    with tf.io.gfile.GFile(bad_file, 'w') as f:
        acc = total = 0
        for qid, ground_truths in answers.items():
            if qid not in predictions:
                total += 1
                continue
            total += 1
            prediction = predictions[qid]
            if prediction == ground_truths:
                acc += 1
            else:
                f.write(json.dumps(samples[qid], ensure_ascii=False) + "|||" + str(prediction) + "\n")
        acc = acc / total
    return {'accuracy': acc}


def main(config: configure_finetuning.FinetuningConfig, split, task_name):
    answers, samples = read_answers(os.path.join(config.raw_data_dir(task_name), split + ".json"))
    predictions = read_predictions(config.qa_preds_file(task_name + "_" + split))
    return evaluate(answers, predictions, samples, config.pred_bad_file(task_name + "_" + split))
