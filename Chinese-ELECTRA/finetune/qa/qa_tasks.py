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

"""Question answering tasks. SQuAD 1.1/2.0 and 2019 MRQA tasks are supported."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import json
import os
import six
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from finetune.qa import qa_metrics
from model import modeling
from model import tokenization
from util import utils
from functools import reduce

class QAExample(task.Example):
    """Question-answering example."""

    def __init__(self,
                 task_name,
                 eid,
                 qas_id,
                 qid,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        super(QAExample, self).__init__(task_name)
        self.eid = eid
        self.qas_id = qas_id
        self.qid = qid
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.start_position:
            s += ", is_impossible: %r" % self.is_impossible
        return s


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def is_whitespace(c):
    return c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(cp)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


class QATask(task.Task):
    """A span-based question answering tasks (e.g., SQuAD)."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, config: configure_finetuning.FinetuningConfig, name,
                 tokenizer, v2=False):
        super(QATask, self).__init__(config, name)
        self._tokenizer = tokenizer
        self._examples = {}
        self.v2 = v2

    def _add_examples(self, examples, example_failures, paragraph, split):
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        if self.name in ["drcd", "cmrc2018"]:  # for chinese
            prev_is_chinese = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace or prev_is_chinese or is_chinese_char(c):
                        doc_tokens.append(c)
                        prev_is_chinese = True if is_chinese_char(c) else False
                    else:
                        doc_tokens[-1] += c
                        prev_is_chinese = False
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
        else:
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in paragraph["qas"]:
            qas_id = qa["id"] if "id" in qa else None
            qid = qa["qid"] if "qid" in qa else None
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            if split == "train":
                if self.v2:
                    is_impossible = qa["is_impossible"]
                if not is_impossible:
                    if "detected_answers" in qa:  # MRQA format
                        answer = qa["detected_answers"][0]
                        answer_offset = answer["char_spans"][0][0]
                    else:  # SQuAD format
                        answer = qa["answers"][0]
                        answer_offset = answer["answer_start"]
                    orig_answer_text = answer["text"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    if answer_offset + answer_length - 1 >= len(char_to_word_offset):
                        utils.log("End position is out of document!")
                        example_failures[0] += 1
                        continue
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]

                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    if self.name in ["drcd", "cmrc2018"]:  # for chinese, no whitespace needed
                        actual_text = "".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                    else:
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                    actual_text = actual_text.lower()
                    cleaned_answer_text = cleaned_answer_text.lower()
                    if actual_text.find(cleaned_answer_text) == -1:
                        utils.log("Could not find answer: '{:}' in doc vs. "
                                  "'{:}' in provided answer".format(
                            tokenization.printable_text(actual_text),
                            tokenization.printable_text(cleaned_answer_text)))
                        example_failures[0] += 1
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = QAExample(
                task_name=self.name,
                eid=len(examples),
                qas_id=qas_id,
                qid=qid,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)
            examples.append(example)

    def get_feature_specs(self):
        return [
            feature_spec.FeatureSpec(self.name + "_eid", []),
            feature_spec.FeatureSpec(self.name + "_start_positions", []),
            feature_spec.FeatureSpec(self.name + "_end_positions", []),
            feature_spec.FeatureSpec(self.name + "_is_impossible", []),
        ]

    def featurize(self, example: QAExample, is_training, log=False,
                  for_eval=False):
        all_features = []
        query_tokens = self._tokenizer.tokenize(example.question_text)

        if len(query_tokens) > self.config.max_query_length:
            query_tokens = query_tokens[0:self.config.max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, self._tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self.config.max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.config.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.config.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.config.max_seq_length
            assert len(input_mask) == self.config.max_seq_length
            assert len(segment_ids) == self.config.max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if log:
                utils.log("*** Example ***")
                utils.log("doc_span_index: %s" % doc_span_index)
                utils.log("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                utils.log("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                utils.log("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                utils.log("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                utils.log("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                utils.log("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    utils.log("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    utils.log("start_position: %d" % start_position)
                    utils.log("end_position: %d" % end_position)
                    utils.log("answer: %s" % (tokenization.printable_text(answer_text)))

            features = {
                "task_id": self.config.task_names.index(self.name),
                self.name + "_eid": (1000 * example.eid) + doc_span_index,
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            if for_eval:
                features.update({
                    self.name + "_doc_span_index": doc_span_index,
                    self.name + "_tokens": tokens,
                    self.name + "_token_to_orig_map": token_to_orig_map,
                    self.name + "_token_is_max_context": token_is_max_context,
                })
            if is_training:
                features.update({
                    self.name + "_start_positions": start_position,
                    self.name + "_end_positions": end_position,
                    self.name + "_is_impossible": example.is_impossible
                })
            all_features.append(features)
        return all_features

    def get_prediction_module(self, bert_model, features, is_training,
                              percent_done):
        final_hidden = bert_model.get_sequence_output()

        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]

        answer_mask = tf.cast(features["input_mask"], tf.float32)
        answer_mask *= tf.cast(features["segment_ids"], tf.float32)
        answer_mask += tf.one_hot(0, seq_length)

        start_logits = tf.squeeze(tf.layers.dense(final_hidden, 1), -1)

        start_top_log_probs = tf.zeros([batch_size, self.config.beam_size])
        start_top_index = tf.zeros([batch_size, self.config.beam_size], tf.int32)
        end_top_log_probs = tf.zeros([batch_size, self.config.beam_size,
                                      self.config.beam_size])
        end_top_index = tf.zeros([batch_size, self.config.beam_size,
                                  self.config.beam_size], tf.int32)
        if self.config.joint_prediction:
            start_logits += 1000.0 * (answer_mask - 1)
            start_log_probs = tf.nn.log_softmax(start_logits)
            start_top_log_probs, start_top_index = tf.nn.top_k(
                start_log_probs, k=self.config.beam_size)

            if not is_training:
                # batch, beam, length, hidden
                end_features = tf.tile(tf.expand_dims(final_hidden, 1),
                                       [1, self.config.beam_size, 1, 1])
                # batch, beam, length
                start_index = tf.one_hot(start_top_index,
                                         depth=seq_length, axis=-1, dtype=tf.float32)
                # batch, beam, hidden
                start_features = tf.reduce_sum(
                    tf.expand_dims(final_hidden, 1) *
                    tf.expand_dims(start_index, -1), axis=-2)
                # batch, beam, length, hidden
                start_features = tf.tile(tf.expand_dims(start_features, 2),
                                         [1, 1, seq_length, 1])
            else:
                start_index = tf.one_hot(
                    features[self.name + "_start_positions"], depth=seq_length,
                    axis=-1, dtype=tf.float32)
                start_features = tf.reduce_sum(tf.expand_dims(start_index, -1) *
                                               final_hidden, axis=1)
                start_features = tf.tile(tf.expand_dims(start_features, 1),
                                         [1, seq_length, 1])
                end_features = final_hidden

            final_repr = tf.concat([start_features, end_features], -1)
            final_repr = tf.layers.dense(final_repr, 512, activation=modeling.gelu,
                                         name="qa_hidden")
            # batch, beam, length (batch, length when training)
            end_logits = tf.squeeze(tf.layers.dense(final_repr, 1), -1,
                                    name="qa_logits")
            if is_training:
                end_logits += 1000.0 * (answer_mask - 1)
            else:
                end_logits += tf.expand_dims(1000.0 * (answer_mask - 1), 1)

            if not is_training:
                end_log_probs = tf.nn.log_softmax(end_logits)
                end_top_log_probs, end_top_index = tf.nn.top_k(
                    end_log_probs, k=self.config.beam_size)
                end_logits = tf.zeros([batch_size, seq_length])
        else:
            end_logits = tf.squeeze(tf.layers.dense(final_hidden, 1), -1)
            start_logits += 1000.0 * (answer_mask - 1)
            end_logits += 1000.0 * (answer_mask - 1)

        def compute_loss(logits, positions):
            one_hot_positions = tf.one_hot(
                positions, depth=seq_length, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
            return loss

        start_positions = features[self.name + "_start_positions"]
        end_positions = features[self.name + "_end_positions"]

        start_loss = compute_loss(start_logits, start_positions)
        end_loss = compute_loss(end_logits, end_positions)

        losses = (start_loss + end_loss) / 2.0

        answerable_logit = tf.zeros([batch_size])
        if self.config.answerable_classifier:
            final_repr = final_hidden[:, 0]
            if self.config.answerable_uses_start_logits:
                start_p = tf.nn.softmax(start_logits)
                start_feature = tf.reduce_sum(tf.expand_dims(start_p, -1) *
                                              final_hidden, axis=1)
                final_repr = tf.concat([final_repr, start_feature], -1)
                final_repr = tf.layers.dense(final_repr, 512,
                                             activation=modeling.gelu)
            answerable_logit = tf.squeeze(tf.layers.dense(final_repr, 1), -1)
            answerable_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(features[self.name + "_is_impossible"], tf.float32),
                logits=answerable_logit)
            losses += answerable_loss * self.config.answerable_weight

        return losses, dict(
            loss=losses,
            start_logits=start_logits,
            end_logits=end_logits,
            answerable_logit=answerable_logit,
            start_positions=features[self.name + "_start_positions"],
            end_positions=features[self.name + "_end_positions"],
            start_top_log_probs=start_top_log_probs,
            start_top_index=start_top_index,
            end_top_log_probs=end_top_log_probs,
            end_top_index=end_top_index,
            eid=features[self.name + "_eid"],
        )

    def get_scorer(self, split="dev"):
        return qa_metrics.SpanBasedQAScorer(self.config, self, split, self.v2)


class MRQATask(QATask):
    """Class for finetuning tasks from the 2019 MRQA shared task."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, name,
                 tokenizer):
        super(MRQATask, self).__init__(config, name, tokenizer)

    def get_examples(self, split):
        if split in self._examples:
            utils.log("N EXAMPLES", split, len(self._examples[split]))
            return self._examples[split]

        examples = []
        example_failures = [0]
        with tf.io.gfile.GFile(os.path.join(
                self.config.raw_data_dir(self.name), split + ".jsonl"), "r") as f:
            for i, line in enumerate(f):
                if self.config.debug and i > 10:
                    break
                paragraph = json.loads(line.strip())
                if "header" in paragraph:
                    continue
                self._add_examples(examples, example_failures, paragraph, split)
        self._examples[split] = examples
        utils.log("{:} examples created, {:} failures".format(
            len(examples), example_failures[0]))
        return examples

    def get_scorer(self, split="dev"):
        return qa_metrics.SpanBasedQAScorer(self.config, self, split, self.v2)


class SQuADTask(QATask):
    """Class for finetuning on SQuAD 2.0 or 1.1."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, name,
                 tokenizer, v2=False):
        super(SQuADTask, self).__init__(config, name, tokenizer, v2=v2)

    def get_examples(self, split):
        if split in self._examples:
            return self._examples[split]

        with tf.io.gfile.GFile(os.path.join(
                self.config.raw_data_dir(self.name),
                split + ("-debug" if self.config.debug else "") + ".json"), "r") as f:
            input_data = json.load(f)["data"]

        examples = []
        example_failures = [0]
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                self._add_examples(examples, example_failures, paragraph, split)
        self._examples[split] = examples
        utils.log("{:} examples created, {:} failures".format(
            len(examples), example_failures[0]))
        return examples

    def get_scorer(self, split="dev"):
        return qa_metrics.SpanBasedQAScorer(self.config, self, split, self.v2)


class SQuAD(SQuADTask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(SQuAD, self).__init__(config, "squad", tokenizer, v2=True)


class SQuADv1(SQuADTask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(SQuADv1, self).__init__(config, "squadv1", tokenizer)


class NewsQA(MRQATask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(NewsQA, self).__init__(config, "newsqa", tokenizer)


class NaturalQuestions(MRQATask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(NaturalQuestions, self).__init__(config, "naturalqs", tokenizer)


class SearchQA(MRQATask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(SearchQA, self).__init__(config, "searchqa", tokenizer)


class TriviaQA(MRQATask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(TriviaQA, self).__init__(config, "triviaqa", tokenizer)


class CMRC2018(SQuADTask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(CMRC2018, self).__init__(config, "cmrc2018", tokenizer)


class DRCD(SQuADTask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(DRCD, self).__init__(config, "drcd", tokenizer)


class MQAExample(task.Example):
    """Multi-choice reading comprehension example."""

    def __init__(self,
                 task_name,
                 eid,
                 qas_id,
                 qid,
                 question_text,
                 _type,  # '0'表示单选题，'1'表示组合单选题, '2'表示多选题
                 options,  # {'A': 'xx', ...} or {'I': 'xx', ...}
                 combination_options=None,  # {'A': ['I', 'II'], ...}
                 evidences=None,  # {'A': ['xx top1', 'xx top2'], ...} or {'I': ['xx top1', 'xx top2'], ...}
                 answers=None):  # ['A']
        super(MQAExample, self).__init__(task_name)
        self.eid = eid
        self.qas_id = qas_id
        self.qid = qid
        self.question_text = question_text
        self.type = _type
        self.options = options
        self.combination_options = combination_options
        self.evidences = evidences
        self.answers = answers

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", options: [%s]" % (" ".join(self.options))
        if self.combination_options:
            s += ", combination_options: %s" % self.combination_options
        if self.evidences:
            s += ", evidences: %s" % self.evidences
        if self.answers:
            s += ", answers: %s" % self.answers
        return s


class MQATask(task.Task):
    """A Multi-choice reading comprehension task."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, config: configure_finetuning.FinetuningConfig, name,
                 tokenizer, v2=False):
        super(MQATask, self).__init__(config, name)
        self._tokenizer = tokenizer
        self._examples = {}
        self.v2 = v2

    def _add_examples(self, examples, example_failures, sample, split):
        question_text = sample["question"]
        _type = sample["type"]
        options = sample["options"]
        combination_options = sample.get("combination_options", None)
        combination_options = combination_options if combination_options else None
        evidences = sample["evidences"]
        qas_id = sample["key"] if "key" in sample else None
        qid = sample["qid"] if "qid" in sample else None
        # data check
        options_keys = options.keys()
        evidences_keys = evidences.keys()
        for _, e_list in evidences.items():
            assert len(e_list) == self.config.evidences_top_k
        assert set(options_keys) == set(evidences_keys)
        if len(options_keys) > self.config.max_options_num:  # not support
            print(sample)
            return
            # if split == "train":  # pass the example if training
            #     return
            # else:  # set useless example if not training
            #     example = MQAExample(task_name=self.name,
            #                          eid=len(examples),
            #                          qas_id=qas_id,
            #                          qid=qid,
            #                          question_text="",
            #                          _type="0",
            #                          options={"C": ""},
            #                          combination_options=None,
            #                          evidences={"C": [""] * self.config.evidences_top_k},
            #                          answers=None)
            #     examples.append(example)
            #     return
        if _type == "0" or _type == "2":
            assert combination_options is None
        elif _type == "1":  # 组合单选题
            assert combination_options is not None

        answers = None
        if split == "train":
            answers = sample.get("answers", None)
            answers = answers if answers else None
            if not answers:  # no label
                example_failures[0] += 1
                return
            # answer check
            if _type == "0":
                assert len(answers) == 1
                assert answers[0] in options_keys
                assert set(options_keys) == set(self.config.answer_options)
            elif _type == "1":
                assert len(answers) == 1
                combination_options_keys = combination_options.keys()
                assert answers[0] in combination_options_keys
                assert set(combination_options_keys) == set(self.config.answer_options)
                for k, comb_ops in combination_options.items():
                    assert len(set(comb_ops)) == len(comb_ops)
                    for op in comb_ops:
                        assert op in options_keys
            else:
                raise Exception("Not implemented for _type not in ('0', '1').")

        example = MQAExample(task_name=self.name,
                             eid=len(examples),
                             qas_id=qas_id,
                             qid=qid,
                             question_text=question_text,
                             _type=_type,
                             options=options,
                             combination_options=combination_options,
                             evidences=evidences,
                             answers=answers)
        examples.append(example)

    def get_feature_specs(self):
        shape = [self.config.max_options_num * self.config.evidences_top_k * self.config.max_seq_length]
        return [
            feature_spec.FeatureSpec("input_ids", shape),
            feature_spec.FeatureSpec("input_mask", shape),
            feature_spec.FeatureSpec("segment_ids", shape),
            feature_spec.FeatureSpec("task_id", []),
            feature_spec.FeatureSpec(self.name + "_eid", []),
            feature_spec.FeatureSpec(self.name + "_answer_mask", [2 ** self.config.max_options_num]),
            feature_spec.FeatureSpec(self.name + "_answer_ids", [2 ** self.config.max_options_num]),
        ]

    def featurize(self, example: MQAExample, is_training, log=False,
                  for_eval=False):
        tokens = []
        input_ids = []
        segment_ids = []
        input_mask = []
        question_tokens = self._tokenizer.tokenize(example.question_text)
        if len(question_tokens) > self.config.max_len1:
            question_tokens = question_tokens[0: self.config.max_len1]
        options_tags = sorted(example.options)
        for op in options_tags:
            op_info = example.options[op]
            op_info_tokens = self._tokenizer.tokenize(op_info)
            if len(op_info_tokens) > self.config.max_len2:
                op_info_tokens = op_info_tokens[0: self.config.max_len2]
            for ev in example.evidences[op]:
                ev_tokens = self._tokenizer.tokenize(ev)
                if len(ev_tokens) > self.config.max_len3 - 1:
                    ev_tokens = ev_tokens[0:(self.config.max_seq_length - len(op_info_tokens) - len(question_tokens) - 4)]
                _tokens = []
                _segment_ids = []
                _tokens.append("[CLS]")
                _segment_ids.append(0)
                for t in question_tokens:
                    _tokens.append(t)
                    _segment_ids.append(0)
                _tokens.append("[SEP]")
                _segment_ids.append(0)
                for t in op_info_tokens:
                    _tokens.append(t)
                    _segment_ids.append(0)
                _tokens.append("[SEP]")
                _segment_ids.append(0)
                for t in ev_tokens:
                    _tokens.append(t)
                    _segment_ids.append(1)
                _tokens.append("[SEP]")
                _segment_ids.append(1)

                _input_ids = self._tokenizer.convert_tokens_to_ids(_tokens)
                _input_mask = [1] * len(_input_ids)

                while len(_input_ids) < self.config.max_seq_length:
                    _input_ids.append(0)
                    _input_mask.append(0)
                    _segment_ids.append(0)

                assert len(_input_ids) == self.config.max_seq_length
                assert len(_input_mask) == self.config.max_seq_length
                assert len(_segment_ids) == self.config.max_seq_length
                tokens.append(_tokens)
                input_ids.append(_input_ids)
                input_mask.append(_input_mask)
                segment_ids.append(_segment_ids)

        # padding for max options number, it may be used in "combination" case.
        padding_num = self.config.max_options_num - len(options_tags)
        while padding_num:
            for _ in range(self.config.evidences_top_k):
                input_ids.append([0] * self.config.max_seq_length)
                input_mask.append([0] * self.config.max_seq_length)
                segment_ids.append([0] * self.config.max_seq_length)
            padding_num -= 1

        answer_ids = None
        if example.type == "0":
            answer_mask = [0] * (2 ** self.config.max_options_num)
            for i in range(len(options_tags)):
                answer_mask[2 ** i] = 1
            if is_training:
                answer_ids = [0] * (2 ** self.config.max_options_num)
                answer_ids[2 ** (options_tags.index(example.answers[0]))] = 1

        elif example.type == "1":
            answer_mask = [0] * (2 ** self.config.max_options_num)
            for _, comb_ops in example.combination_options.items():
                index = 0
                for comb_op in comb_ops:
                    index += 2 ** (options_tags.index(comb_op))
                answer_mask[index] = 1
            if is_training:
                answer_ids = [0] * (2 ** self.config.max_options_num)
                index = 0
                for comb_op in example.combination_options[example.answers[0]]:
                    index += 2 ** (options_tags.index(comb_op))
                answer_ids[index] = 1
        else:
            raise Exception("Not implemented for _type not in ('0', '1').")

        # flat
        def flat(x):
            return reduce(lambda a, b: a+b, x)
        tokens = flat(tokens)
        input_ids = flat(input_ids)
        input_mask = flat(input_mask)
        segment_ids = flat(segment_ids)

        if log:
            utils.log("*** Example ***")
            utils.log("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            utils.log("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            utils.log("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            utils.log("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            utils.log("answer: %s" % " ".join([str(x) for x in answer_mask]))
            if is_training:
                utils.log("answer: %s" % " ".join([str(x) for x in answer_ids]))

        features = {
            "task_id": self.config.task_names.index(self.name),
            self.name + "_eid": example.eid,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            self.name + "_answer_mask": answer_mask,
        }
        if for_eval:
            features.update({
                self.name + "_options_tags": options_tags,
                self.name + "_combination_options": example.combination_options,
                self.name + "_type": example.type,
            })
        if is_training:
            features.update({
                self.name + "_answer_ids": answer_ids,
            })

        return [features]

    def get_prediction_module(self, bert_model, features, is_training,
                              percent_done):
        final_hidden = bert_model.get_pooled_output()
        # bs * options_num * top_k, hidden_dim
        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=2)
        # bs, options_num * top_k * seq_len
        input_ids_shape = modeling.get_shape_list(features["input_ids"], expected_rank=2)
        batch_size = input_ids_shape[0]
        hidden_dim = final_hidden_shape[1]
        final_hidden_reshape = tf.reshape(final_hidden, [batch_size, self.config.max_options_num,
                                                         self.config.evidences_top_k * hidden_dim])

        # def single(hidden, mask, y):
        #     logits = tf.squeeze(tf.layers.dense(hidden, 1), -1)
        #     mask = mask[:self.config.max_options_num]
        #     y = y[:self.config.max_options_num]
        #     logits_masked = logits + 1e8 * (mask - 1)
        #     loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_masked)
        #     return logits_masked, loss
        #
        # def combination_single(hidden, mask, y):
        #     logits = tf.squeeze(tf.layers.dense(hidden, 1), -1)  # todo: share or not ?
        #     logits = tf.layers.dense(logits, 2 ** self.config.max_options_num)
        #     logits_masked = logits + 1e8 * (mask - 1)
        #     loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_masked)
        #     return logits_masked, loss
        # logits, loss = tf.cond()

        logits = tf.squeeze(tf.layers.dense(final_hidden_reshape, 1), -1)
        logits = tf.layers.dense(logits, 2 ** self.config.max_options_num)
        logits_masked = logits + 1e8 * tf.to_float(features[self.name + "_answer_mask"] - 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=features[self.name + "_answer_ids"], logits=logits_masked)

        return loss, dict(
            loss=loss,
            logits=logits_masked,
            eid=features[self.name + "_eid"],
        )

    def get_scorer(self, split="dev"):
        return qa_metrics.MQAScorer(self.config, self, split, self.v2)


class SAC(MQATask):
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
        super(SAC, self).__init__(config, "sac", tokenizer)

    def get_examples(self, split):
        if split in self._examples:
            return self._examples[split]

        input_data = []
        with tf.io.gfile.GFile(os.path.join(
                self.config.raw_data_dir(self.name),
                split + ("-debug" if self.config.debug else "") + ".json"), "r") as f:
            for line in f:
                input_data.append(json.loads(line))

        examples = []
        example_failures = [0]
        for sample in input_data:
            self._add_examples(examples, example_failures, sample, split)
        self._examples[split] = examples
        utils.log("{:} examples created, {:} failures".format(
            len(examples), example_failures[0]))
        return examples
