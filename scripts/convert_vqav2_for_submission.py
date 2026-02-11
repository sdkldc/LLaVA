import os
import argparse
import json

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./playground/data/eval/vqav2")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    src = os.path.join(args.dir, 'answers', args.split, args.ckpt, 'merge.jsonl')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Missing merged answers: {src}\n"
            f"Run inference first (e.g., scripts/v1_5/eval/vqav2.sh) to create merge.jsonl."
        )
    test_split = os.path.join(args.dir, f'{args.split}.jsonl')
    if not os.path.exists(test_split):
        legacy_test_split = os.path.join(args.dir, 'llava_vqav2_mscoco_test2015.jsonl')
        if os.path.exists(legacy_test_split):
            test_split = legacy_test_split
        else:
            raise FileNotFoundError(
                f"Missing question file: {test_split}\n"
                f"Expected under: {args.dir}\n"
                f"If you only have VQA v2 test-dev, set --split llava_vqav2_mscoco_test-dev2015 "
                f"and ensure {os.path.join(args.dir, 'llava_vqav2_mscoco_test-dev2015.jsonl')} exists."
            )
    dst = os.path.join(args.dir, 'answers_upload', args.split, f'{args.ckpt}.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    results = []
    error_line = 0
    for _, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(results[x['question_id']])
            })

    with open(dst, 'w') as f:
        json.dump(all_answers, f)
    print(f"wrote: {dst}")
