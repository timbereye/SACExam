# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Official evaluation script for v1.1 of the SQuAD dataset.
Modified slightly for the ELECTRA codebase.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter
import string
import re
import json
import sys
import os
import collections
import tensorflow.compat.v1 as tf

import configure_finetuning


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def main(config: configure_finetuning.FinetuningConfig, split, task_name="squadv1"):
    expected_version = '1.1'
    # parser = argparse.ArgumentParser(
    #     description='Evaluation for SQuAD ' + expected_version)
    # parser.add_argument('dataset_file', help='Dataset file')
    # parser.add_argument('prediction_file', help='Prediction File')
    # args = parser.parse_args()
    Args = collections.namedtuple("Args", [
        "dataset_file", "prediction_file"
    ])
    args = Args(dataset_file=os.path.join(
        config.raw_data_dir(task_name),
        split + ("-debug" if config.debug else "") + ".json"),
        prediction_file=config.qa_preds_file(task_name))
    with tf.io.gfile.GFile(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with tf.io.gfile.GFile(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    return evaluate(dataset, predictions)
