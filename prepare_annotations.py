"""Export one human answer-label row per generated question/answer pair."""

import argparse
import csv


ANNOTATION_FIELDS = [
    "example_index", "question", "generated_text",
    "human_label", "annotator_notes",
]
HUMAN_LABELS = {"supported", "hallucinated", "abstain"}


def load_answer_groups(data_path: str) -> list[dict]:
    answers = {}
    with open(data_path, newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            required = {"example_index", "question", "generated_text"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"Feature data is missing columns: {sorted(missing)}")
            example_id = row["example_index"]
            answer = (row["question"], row["generated_text"])
            if example_id not in answers:
                answers[example_id] = {
                    "example_index": example_id,
                    "question": answer[0],
                    "generated_text": answer[1],
                    "token_rows": [],
                    "machine_labels": set(),
                }
            elif (answers[example_id]["question"], answers[example_id]["generated_text"]) != answer:
                raise ValueError(
                    f"Example {example_id!r} has inconsistent question/answer rows"
                )
            answers[example_id]["token_rows"].append(row)
            if row.get("label", "") != "":
                answers[example_id]["machine_labels"].add(str(row["label"]))

    if not answers:
        raise ValueError(f"No feature rows found in {data_path}")
    for answer in answers.values():
        if len(answer["machine_labels"]) > 1:
            raise ValueError(
                f"Example {answer['example_index']!r} has inconsistent automatic labels"
            )
        answer["machine_label"] = next(iter(answer["machine_labels"]), "")
    return list(answers.values())


def export_annotation_template(data_path: str, output_path: str) -> int:
    answers = load_answer_groups(data_path)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for answer in answers:
            writer.writerow({
                "example_index": answer["example_index"],
                "question": answer["question"],
                "generated_text": answer["generated_text"],
                "human_label": "",
                "annotator_notes": "",
            })

    return len(answers)


def main():
    parser = argparse.ArgumentParser(
        description="Create an answer-level human annotation template from token features."
    )
    parser.add_argument("--data", default="results_tokens.csv")
    parser.add_argument("--out", default="human_labels.csv")
    args = parser.parse_args()
    answer_count = export_annotation_template(args.data, args.out)
    print(f"Wrote {answer_count} answer annotation rows to {args.out}")
    print("Fill human_label with supported, hallucinated, or abstain.")


if __name__ == "__main__":
    main()
