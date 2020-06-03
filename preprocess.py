#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     preprocess.py
#
# Description:
# Version:      1.0
# Created:      2020/5/29 14:25
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#

"""
证券从业考试数据集处理，包含两个科目
1. 金融市场基础知识
2. 证券市场基本法律法规
数据集格式为每行一条json，包含一道题目相关的信息，具体包含字段如下：
key 唯一标识
subject 科目，0表示金融市场基础知识，1表示证券市场基本法律法规
source 题目来源，0表示历年真题，1表示模拟题，2表示其他
type 题目类型，0表示单选题，1表示组合单选题，2表示多选题（2016年开始多选题改成组合单选题，判断题被废弃）
question 问题描述
options 选项，对于单选题：e.g {"A":XXX, "B":XXX, ...} , 对于多选题：e.g {"I":XXX, "II":XXX, ...} 这里是用于组合的选项，键值不定
combination_options 组合选项，这是组合单选题真正的选项，e.g {"A":["I", "II"], "B":["I", "IV"], ...}
answers 答案，e.g ["A"]
analysis 解析
"""

import glob
import json
import os
import re
import uuid

SUBJECTS = {"financial": "0", "law": "1", "unknown": "2"}
SOURCES = {"true": "0", "mock": "1"}
TYPES = {"single": "0", "combination": "1", "multi": "2"}

JSON_DATA = {"key":"", "subject":"", "source":"", "type":"", "question":"", "options":{},
             "combination_options":{}, "answers":[], "analysis": ""}

ROMAN = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ"]


def parse_options_lines(lines):
    if re.match(r"A[.、,，]", lines[0]): # 单选题
        question_type = "single"
        options = {}
        for line in lines:
            matcher = re.match(r"([ABCD])[.、,，](.*)", line)
            if matcher:
                option_abc = matcher.group(1)
                option_content = matcher.group(2)
                options[option_abc] = option_content
            else:
                options[option_abc] += line  # multi-line option
        # assert len(options) == 4, "options error: {}, {}".format(lines, options)
        if len(options) != 4:  # ABCD缺失
            print(lines, options)
            return None, None, None
        return question_type, options, None
    elif re.match(r"Ⅰ[.、,，]", lines[0]):  # 单项组合选择题
        question_type = "combination"
        options = {}
        combination_options = {}
        text_op = ""
        lines_comb = ""
        for i, line in enumerate(lines):
            if not re.match(r"A[.、,，]", line):
                text_op += line
            else:
                lines_comb = lines[i:]
                break
        re_exp_op = re.compile("([{0}])[.、,，]([^{0}]*)".format("".join(ROMAN)))
        for option_roman, option_content in re.findall(re_exp_op, text_op):
            options[option_roman] = option_content
        assert len(options) >= 3, lines
        for line in lines_comb:
            matcher = re.match(r"([ABCD])[.、,，](.*)", line)
            if matcher:
                option_abc = matcher.group(1)
                option_content = matcher.group(2)
                option_contents = re.split(r"[.、,，;；]", option_content)
                option_contents = [x for x in option_contents if x]
                flag = False  # 检查罗马数字是否有误，丢弃脏数据
                for i, oc in enumerate(option_contents):
                    if oc not in ROMAN:
                        if oc == "I":  # fix error
                            option_contents[i] = "Ⅰ"
                        else:
                            flag = True
                            print(lines)
                            print(option_abc)
                            print(option_content)
                            print(option_contents)
                            break
                if flag:
                    return None, None, None
                combination_options[option_abc] = option_contents

        if len(combination_options) != 4:  # ABCD缺失
            print(lines, combination_options)
            return None, None, None
        return question_type, options, combination_options


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        # 全角区间
        if inside_code >= 0xFF01 and inside_code <= 0xFF5E:
            inside_code -= 0xfee0
            rstring += chr(inside_code)
        # 全角空格特殊处理
        elif inside_code == 0x3000 or inside_code ==0x00A0:
            inside_code = 0x0020
            rstring += chr(inside_code)
        else:
            rstring += uchar
    return rstring


def process(input_file_pattern, subject="financial", source="true"):
    for f in glob.iglob(str(input_file_pattern)):
        if not os.path.isfile(f):
            continue
        print(f)
        with open(f, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            lines = [re.sub(r"\s+|&nbsp;|<br/>|&emsp;|<p>|</p>", "", line) for line in lines]  # 去掉空白和脏字符
            lines = [strQ2B(line) for line in lines if len(line) > 2 and not re.match(r"^\d+/\d+$", line)]  # 过滤页码
            line_no = 0
            question_index = 1
            # 由于选项信息很复杂，我们先不解析选项
            examples = []
            while line_no < len(lines):
                # question
                while line_no < len(lines):
                    if re.match(re.compile("{0}[、.,，]".format(str(question_index))), lines[line_no]):
                        question = lines[line_no].lstrip(str(question_index)+"、")
                        line_no += 1
                        while line_no < len(lines):
                            if not re.match(r"[AⅠ][.、,，]", lines[line_no]):
                                question += lines[line_no]
                                line_no += 1
                            else:
                                break
                        break
                    else:
                        line_no += 1
                # get answer and cache options info
                options_lines = []
                while line_no < len(lines):
                    if re.match((r"答案[:：]"), lines[line_no]):
                        answer = re.search(r"[A-D]+", lines[line_no]).group()
                        assert len(answer) == 1
                        answer = [answer]
                        line_no += 1
                        break
                    else:
                        options_lines.append(lines[line_no])
                        line_no += 1
                # analysis
                while line_no < len(lines):
                    if re.match((r"解析[:：]"), lines[line_no]):
                        analysis = re.sub(r"^解析[:：]", "", lines[line_no])
                        line_no += 1
                        while line_no < len(lines):
                            if not re.match(re.compile("{0}[、.,，]".format(str(question_index + 1))), lines[line_no]):
                                analysis += lines[line_no]
                                line_no += 1
                            else:
                                break
                        break
                    else:
                        line_no += 1
                assert len(question) > 0
                examples.append({"question": question, "answers": answer, "analysis": analysis if len(analysis) > 5 else "",
                                 "options_lines": options_lines})
                question_index += 1

            # 解析选项信息
            examples_new = []
            for example in examples:
                options_lines = example["options_lines"]
                example.pop("options_lines")
                question_type, options, combination_options = parse_options_lines(options_lines)
                if question_type == "single":
                    example["options"] = options
                    example["combination_options"] = {}
                elif question_type == "combination":
                    example["options"] = options
                    example["combination_options"] = combination_options
                elif question_type is None:  # pass
                    continue
                example["type"] = TYPES.get(question_type)
                example["subject"] = SUBJECTS.get(subject)
                example["source"] = SOURCES.get(source)
                example["key"] = "".join(str(uuid.uuid1()).split("-"))
                examples_new.append(example)
            print(len(examples_new))
            yield examples_new


def check_reduplicate(examples):
    examples_new = []
    infos = set()
    for example in examples:
        info = example["question"] + str(sorted(example["options"].items(), key=lambda x:x[0])) +\
               str(sorted(example["combination_options"].items(), key=lambda x:x[0]))
        if info not in infos:
            infos.add(info)
            examples_new.append(example)
        # else:
        #     print(info)
    return examples_new


def maomin_data_process(input_file_pattern):
    for f in glob.iglob(input_file_pattern):
        print(f)
        with open(f, 'r', encoding="utf-8") as fp, open("tmp.txt", 'w', encoding="utf-8") as ft:
            data = []
            for line in fp:
                if "img" in line or "http" in line:
                    print(line)
                    continue
                data.append(json.loads(line))
            # 构造成试卷格式以复用处理代码, 以及清洗数据
            for i, d in enumerate(data):
                tmp = ["{}、{}".format(i+1, re.sub(r"91速过.*", "", d["query"].replace("\r\n", "\n").replace("Ⅰ", "\nⅠ"))),
                       re.sub(r"([ABCD])([、.,，:])", r"\n\1、", re.sub(r"&nbsp;|<br/>", "", d["candidates"])),
                       "答案：{}".format(d["std_an"]),
                       "解析：{}".format(re.sub(r"^【[答案及和与解析]+】", "", re.sub(r"\s+|91速过.*", "", d["an_parse"])))]
                text = "\n".join(tmp)
                ft.write(text + "\n")
        examples = process("tmp.txt", subject="unknown", source="mock")
        return examples




def main():
    input_file_pattern = "data/financial/mock/*.txt"
    examples1 = process(input_file_pattern, subject="financial", source="mock")
    input_file_pattern = "data/financial/true/*.txt"
    examples2 = process(input_file_pattern, subject="financial", source="true")
    input_file_pattern = "data/law/mock/*.txt"
    examples3 = process(input_file_pattern, subject="law", source="mock")
    input_file_pattern = "data/law/true/*.txt"
    examples4 = process(input_file_pattern, subject="law", source="true")

    input_file_pattern = "data/merge.json"
    examples5 = maomin_data_process(input_file_pattern)

    total_examples = []
    for examples in [examples1, examples2, examples3, examples4, examples5]:
        for examples_ in examples:
            total_examples += examples_
    print("total examples:", len(total_examples))

    total_examples = check_reduplicate(total_examples)
    print("unique total examples:", len(total_examples))

    with open("data/data.json", 'w', encoding="utf-8") as fp:
        for ex in total_examples:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
