""" Official evaluation script for v1.1 of the SQuAD dataset. """
import re
import string
import sys
from collections import Counter
import os
import argparse


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

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
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    RW = [[], []]
    all_em = []
    all_f1 = []
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                all_em.append(em)
                if em == 1:
                    RW[0].append(qa['id'])
                else:
                    RW[1].append(qa['id'])
                exact_match += em
                sub_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                all_f1.append(sub_f1)
                f1 += sub_f1


    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}, RW, all_em,all_f1


def compute_metric(predictions, references):
    pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
    dataset = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                            "id": ref["id"],
                        }
                        for ref in references
                    ]
                }
            ]
        }
    ]
    score, RW,all_em,all_f1 = evaluate(dataset=dataset, predictions=pred_dict)
    return score, RW,all_em,all_f1

def further_process(span):
    # select the shortest item in the enumeration, if the same length, select the first

    first_shortest = None

    span = span.replace('\"','')

    if ",and" in span or ", and" or " and ":
        span=span.replace(",and",", ")
        span=span.replace(", and", ", ")
        span=span.replace(" and ", ", ")
    # now no and in, only ", "
    if "," in span:
        entities = span.split(',')
        entities = [ent.strip() for ent in entities]
        first_shortest = entities[0]
    return span if first_shortest is None else first_shortest


if __name__ == '__main__':
    from data_utils import ODQATextData

    parser = argparse.ArgumentParser()
    parser.add_argument('--taskname', default='ablation_triviaqa')
    parser.add_argument('--pred_filename', default=None)
    parser.add_argument('--number_preds', type=int, default=-1)
    parser.add_argument('--dataset_dir', default='../../datasets/nonhf/')
    args = parser.parse_args()

    args.dataset_dir = os.path.join(args.dataset_dir, args.taskname)

    obj = ODQATextData('test', args, eval_only=True)

    predfile = args.pred_filename
    preds_rsa = []

    raw_preds = []

    with open(predfile) as f:
        for id, line in enumerate(f.readlines()):
            ll = line.strip().split('\t')
            preds_rsa.append(further_process(ll[2]))
            raw_preds.append(ll[2])

    print('>>> ', args.taskname)
    print('After process')
    res, _, _, _ = obj.compute_metric(preds_rsa, args.number_preds)
    print(res)

    print('No process')
    res, _, _, _ = obj.compute_metric(raw_preds, args.number_preds)
    print(res)
