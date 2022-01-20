import logging
from typing import Iterable, List
import numpy as np
import json
import string
import os


def word_level_edit_distance(a: List[str], b: List[str]) -> int:
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]


def edit_distance(sent1: List[str], sent2: List[str]) -> int:
    return word_level_edit_distance(sent1, sent2)


def relative_distance(src_ref_dis, hypo_ref_dis):
    if src_ref_dis == 0:
        logging.error("src_ref is the same as ref.")
        src_ref_dis = 1
        if hypo_ref_dis == 0:
            return 1
    return hypo_ref_dis / src_ref_dis


def refine(one_str):
    tran_tab = str.maketrans({key: None for key in string.punctuation})
    new_str = one_str.translate(tran_tab)
    return new_str.lower()


def eval(hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
         src_references: Iterable[List[str]], *args, **kwargs) -> dict:
    src_distances = []
    hypo_distances = []
    rel_distances = []
    cnt = 0
    for idx, (hypo_list, ref, src_ref) in enumerate(zip(hypos, references, src_references)):
        hypo = hypo_list[0]
        hypo_ref_dis = edit_distance(hypo, ref)*1.416
        src_ref_dis = edit_distance(src_ref, ref)*1.416
        src_distances.append(src_ref_dis)
        hypo_distances.append(hypo_ref_dis)
        rel_distances.append(relative_distance(src_ref_dis, hypo_ref_dis))
    rel_dis = float(np.mean(rel_distances))
    src_dis = float(np.mean(src_distances))
    hypo_dis = float(np.mean(hypo_distances))
    return {"rel_distance": rel_dis, "hypo_distance": hypo_dis, "src_distance": src_dis}


def eval_hebcup():
    PREDICTION_DIR = "../hebcup_result"
    src_file = os.path.join(PREDICTION_DIR, "src.txt")
    ref_file = os.path.join(PREDICTION_DIR, "ref.txt")
    pred_file = os.path.join(PREDICTION_DIR, "pred.txt")
    src_tokens = []
    gold_tokens = []
    pred_tokens = []

    with open(src_file, 'r', encoding='utf-8') as src_f:
        for line in src_f.readlines():
            cur_token_list = refine(line.strip()).split()
            src_tokens.append(cur_token_list)

    with open(ref_file, 'r', encoding='utf-8') as gold_f:
        for line in gold_f.readlines():
            cur_token_list = refine(line.strip()).split()
            gold_tokens.append(cur_token_list)
    with open(pred_file, 'r', encoding='utf-8') as pred_f:
        for line in pred_f.readlines():
            pres = json.loads(line.strip())
            cur_token_list = refine(pres[0]).split()
            pred_tokens.append([cur_token_list])

    res = eval(pred_tokens, gold_tokens, src_tokens)
    print(res)


def eval_mine():
    PREDICTION_DIR = "../complex_result"
    src_file = os.path.join(PREDICTION_DIR, "src.txt")
    ref_file = os.path.join(PREDICTION_DIR, "ref.txt")
    pred_file = os.path.join(PREDICTION_DIR, "pred.txt")
    src_tokens = []
    gold_tokens = []
    pred_tokens = []

    with open(src_file, 'r', encoding='utf-8') as src_f:
        for line in src_f.readlines():
            cur_token_list = refine(line.strip()).split()
            src_tokens.append(cur_token_list)

    with open(ref_file, 'r', encoding='utf-8') as gold_f:
        for line in gold_f.readlines():
            cur_token_list = refine(line.strip()).split()
            gold_tokens.append(cur_token_list)
    with open(pred_file, 'r', encoding='utf-8') as pred_f:
        for line in pred_f.readlines():
            pres = json.loads(line.strip())
            cur_token_list = refine(pres[0]).split()
            pred_tokens.append([cur_token_list])

    res = eval(pred_tokens, gold_tokens, src_tokens)
    print(res)


if __name__ == '__main__':
    eval_mine()