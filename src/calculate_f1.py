import json
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-output", type=str, required=True)
    parser.add_argument("--qwen-output", type=str, required=True)
    return parser.parse_args()


def get_tag_list(path, entry="domains"):
    with open(path, "r", encoding="utf-8") as f:
        tag_list = [json.loads(line)[entry] for line in f]
    return tag_list


def tags_F1(tag_list_1, tag_list_2):
    t1, t2 = set(tag_list_1), set(tag_list_2)
    overlap = t1.intersection(t2)
    true_positives = len(overlap)
    
    # Handle edge cases to avoid division by zero
    if len(t1) == 0:
        precision = 0.0
    else:
        precision = true_positives / len(t1)
    
    if len(t2) == 0:
        recall = 0.0
    else:
        recall = true_positives / len(t2)
    
    # Calculate F1 score
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / float(precision + recall)
    return f1


def main(args):
    gpt_tag_lists = get_tag_list(args.gpt_output)
    qwen_tag_lists = get_tag_list(args.qwen_output)
    f1_scores = []
    for t1, t2 in zip(gpt_tag_lists, qwen_tag_lists):
        f1_scores.append(tags_F1(t1, t2))
    print("f1 score is", sum(f1_scores) / len(f1_scores))


if __name__ == "__main__":
    main(parse())