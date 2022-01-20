import json
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from SARI import SARIsent


def compute_sentence_meteor(reference_list, sentences):
    preds = dict()
    refs = dict()

    for i in range(len(sentences)):
        preds[i] = [' '.join([s for s in sentences[i]])]
        refs[i] = [' '.join(l) for l in reference_list[i]]

    final_scores = dict()

    scorers = [
        (Meteor(),"METEOR")
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = scs
        else:
            final_scores[method] = scores

    meteor_scores = final_scores["METEOR"]
    return meteor_scores


def compute_meteor(reference_list, sentences):
    meteor_scores = compute_sentence_meteor(reference_list, sentences)
    return 100 * sum(meteor_scores)/len(meteor_scores)


def compute_sari(source_sentences, target_sentences, predictions):
    target_sentences = [[i] for i in target_sentences]
    predicted_sentences = predictions

    inp = zip(source_sentences, target_sentences, predicted_sentences)
    scores = []
    for source, target, predicted in inp:
        scores.append(SARIsent(source, predicted, target))
    return 100 * sum(scores) / float(len(scores))


def compute_accuracy(reference_strings, predicted_strings):
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    for i in range(len(reference_strings)):
        cur_ref = refine(reference_strings[i])
        cur_pre = refine(predicted_strings[i][0])
        if cur_ref.replace(' ', '') == cur_pre.replace(' ', ''):
            correct += 1
    print("total correct in top1:", int(correct))
    return 100 * correct/float(len(reference_strings))


def compute_accuracy_rerank(src_strings, reference_strings, predicted_strings):
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    count_rerank = 0
    for i in range(len(reference_strings)):
        cur_ref = reference_strings[i]
        if predicted_strings[i][0] == src_strings[i]:
            cur_pre = predicted_strings[i][1]
            count_rerank += 1
            print(i)
            print(src_strings[i])
            print(cur_ref)
            print(predicted_strings[i][0])
        else:
            cur_pre = predicted_strings[i][0]
        if cur_ref.replace(' ', '') == cur_pre.replace(' ', ''):
            correct += 1
    print("total correct in top1:", int(correct))
    print("count_rerank:", count_rerank)
    return 100 * correct/float(len(reference_strings))


def compute_recall(reference_strings, predicted_strings):
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    for i in range(len(reference_strings)):
        cur_idx = 0
        cur_ref = refine(reference_strings[i])
        for sentence in predicted_strings[i]:
            cur_pre = refine(sentence)
            if cur_ref.replace(' ', '') == cur_pre.replace(' ', ''):
                correct += 1
                break
            cur_idx += 1
    print("total correct in top5:", int(correct))
    return 100 * correct/float(len(reference_strings))


def refine(one_str):
    tran_tab = str.maketrans({key: None for key in string.punctuation})
    new_str = one_str.translate(tran_tab)
    return new_str.lower()


def compute_metrics(src_file, ref_file, pred_file):
    src_strs = []
    gold_strs = []
    pred_strs = []
    pred_1_str = []
    gold_tokens = []
    pred_tokens = []

    with open(src_file, 'r', encoding='utf-8') as src_f:
        for line in src_f.readlines():
            src_strs.append(line.strip())

    with open(ref_file, 'r', encoding='utf-8') as gold_f:
        for line in gold_f.readlines():
            gold_strs.append(line.strip())
            refine_str = refine(line.strip())
            gold_tokens.append([refine_str.split()])
    with open(pred_file, 'r', encoding='utf-8') as pred_f:
        for line in pred_f.readlines():
            pres = json.loads(line.strip())
            pred_strs.append(pres)
            pred_1_str.append(pres[0])
            refine_str = refine(pres[0])
            pred_tokens.append(refine_str.split())

    predicted_accuracy = compute_accuracy(gold_strs, pred_strs)
    predicted_recall = compute_recall(gold_strs, pred_strs)
    predicted_meteor = compute_meteor(gold_tokens, pred_tokens)
    predicted_sari = compute_sari(src_strs, gold_strs, pred_1_str)
    print('Predicted Accuracy: {}'.format(predicted_accuracy))
    print('Predicted Recall: {}'.format(predicted_recall))
    print('Predicted Meteor: {}'.format(predicted_meteor))
    print('Predicted SARI: {}'.format(predicted_sari))


if __name__ == '__main__':
    src_file = "../complex_result/src.txt"
    ref_file = '../complex_result/ref.txt'
    pred_file = '../complex_result/pred.txt'
    compute_metrics(src_file, ref_file, pred_file)
