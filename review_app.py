"""Interactive Streamlit review UI for answer-level human annotations."""

import csv
import io
from pathlib import Path

import streamlit as st

from prepare_annotations import ANNOTATION_FIELDS, HUMAN_LABELS, load_answer_groups


def read_annotations(path: str) -> dict[str, dict]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if set(ANNOTATION_FIELDS) - set(reader.fieldnames or ()):
            raise ValueError(
                f"{path} must have columns {', '.join(ANNOTATION_FIELDS)}"
            )
        annotations = {}
        for row in reader:
            example_id = str(row["example_index"])
            if example_id in annotations:
                raise ValueError(f"Duplicate annotation for answer {example_id!r}")
            label = row["human_label"].strip().casefold()
            if label not in HUMAN_LABELS | {"", "uncertain"}:
                raise ValueError(
                    f"Invalid label {label!r} for answer {example_id!r}"
                )
            annotations[example_id] = {
                **row,
                "human_label": "abstain" if label == "uncertain" else label,
            }
    return annotations


def annotation_csv(answers: list[dict], annotations: dict[str, dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=ANNOTATION_FIELDS)
    writer.writeheader()
    for answer in answers:
        annotation = annotations.get(str(answer["example_index"]), {})
        writer.writerow({
            "example_index": answer["example_index"],
            "question": answer["question"],
            "generated_text": answer["generated_text"],
            "human_label": annotation.get("human_label", ""),
            "annotator_notes": annotation.get("annotator_notes", ""),
        })
    return output.getvalue()


def persist_review(example_id: str, label_key: str, notes_key: str, output_path: str):
    label = st.session_state[label_key]
    st.session_state.review_annotations[example_id] = {
        "human_label": {
            "Not reviewed": "",
            "Supported": "supported",
            "Hallucinated": "hallucinated",
            "Abstain": "abstain",
        }[label],
        "annotator_notes": st.session_state.get(notes_key, ""),
    }
    Path(output_path).write_text(
        annotation_csv(
            st.session_state.review_answers,
            st.session_state.review_annotations,
        ),
        encoding="utf-8",
        newline="",
    )
    st.session_state.last_saved_answer = example_id


def heuristic_label(answer: dict) -> tuple[str, str]:
    labels = answer["machine_labels"]
    if not labels:
        return "Not available", "No automatic label found in feature CSV."
    label = next(iter(labels))
    if label == "1":
        verdict = "Hallucinated / incorrect (heuristic)"
    elif label == "0":
        verdict = "Supported / correct (heuristic)"
    else:
        verdict = f"Automatic label: {label}"

    rows = answer["token_rows"]
    correct_overlap = rows[0].get("max_correct_overlap", "")
    incorrect_overlap = rows[0].get("max_incorrect_overlap", "")
    detail = "The previous run's answer-level lexical-overlap heuristic; not a human judgment."
    if correct_overlap != "" or incorrect_overlap != "":
        detail += (
            f" Correct-reference overlap: {correct_overlap or 'n/a'}; "
            f"incorrect-reference overlap: {incorrect_overlap or 'n/a'}."
        )
    return verdict, detail


def token_table_rows(answer: dict) -> list[dict]:
    output = []
    for row in answer["token_rows"]:
        raw_label = str(row.get("label", "")).strip()
        existing_label = {
            "0": "Supported (heuristic)",
            "1": "Hallucinated (heuristic)",
        }.get(raw_label, raw_label)
        output.append({
            "Content token": row.get("token", ""),
            "Evidence span": row.get("evidence_span", ""),
            "Counterfactual ΔlogP": row.get(
                "counterfactual_score", row.get("delta_logprob", "")
            ),
            "Semantic similarity": row.get(
                "semantic_similarity", row.get("evidence_similarity", "")
            ),
            "Existing auto label": existing_label,
        })
    return output


def main():
    st.set_page_config(page_title="Hallucination annotation review", layout="wide")
    st.title("Answer-level hallucination review")
    st.caption(
        "Judge the full response using the question and available factual context. "
        "The automatic label is shown for comparison only."
    )

    with st.sidebar:
        st.header("Files")
        data_path = st.text_input("Feature CSV", value="results_tokens.csv")
        output_path = st.text_input("Human-label CSV", value="human_labels.csv")
        st.caption("Annotations are written to the human-label CSV whenever you change a label or note.")

    try:
        answers = load_answer_groups(data_path)
        annotations = read_annotations(output_path)
    except (OSError, ValueError, csv.Error) as error:
        st.error(f"Could not load review files: {error}")
        st.stop()

    current_ids = {str(answer["example_index"]) for answer in answers}
    orphaned_ids = set(annotations) - current_ids
    if orphaned_ids:
        st.error(
            f"{output_path} contains annotations for IDs not present in {data_path} "
            f"({', '.join(sorted(orphaned_ids)[:5])}). Choose another output file "
            "or restore the matching feature CSV to avoid overwriting annotations."
        )
        st.stop()

    st.session_state.review_answers = answers
    st.session_state.review_annotations = annotations

    def label_for(answer):
        return annotations.get(str(answer["example_index"]), {}).get("human_label", "")

    reviewed = sum(label_for(answer) in HUMAN_LABELS for answer in answers)
    st.progress(reviewed / len(answers), text=f"Reviewed {reviewed} of {len(answers)} answers")

    def answer_option(index):
        answer = answers[index]
        question = " ".join(answer["question"].split())
        if len(question) > 100:
            question = question[:97] + "..."
        label = label_for(answer)
        status = label if label else "unreviewed"
        return f"{index + 1}/{len(answers)} · {status} · {question}"

    selected = st.selectbox(
        "Choose a response",
        options=list(range(len(answers))),
        format_func=answer_option,
        key="selected_answer_index",
    )
    answer = answers[selected]
    example_id = str(answer["example_index"])

    automatic, explanation = heuristic_label(answer)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Question")
        st.write(answer["question"])
    with right:
        st.subheader("Existing automatic claim")
        st.warning(automatic)
        st.caption(explanation)

    st.subheader("Generated response")
    st.markdown(answer["generated_text"])
    st.subheader("Existing per-content-token scores")
    st.dataframe(token_table_rows(answer), use_container_width=True, hide_index=True)

    saved = annotations.get(example_id, {})
    label_key = f"human_label_{example_id}"
    notes_key = f"annotator_notes_{example_id}"
    label_key = f"{data_path}_{output_path}_{label_key}"
    notes_key = f"{data_path}_{output_path}_{notes_key}"
    display_by_label = {
        "": "Not reviewed",
        "supported": "Supported",
        "hallucinated": "Hallucinated",
        "abstain": "Abstain",
    }
    reverse_display = {value: key for key, value in display_by_label.items()}
    if label_key not in st.session_state:
        st.session_state[label_key] = display_by_label.get(
            saved.get("human_label", ""), "Not reviewed"
        )
    if notes_key not in st.session_state:
        st.session_state[notes_key] = saved.get("annotator_notes", "")
    callback = persist_review
    callback_args = (example_id, label_key, notes_key, output_path)

    st.radio(
        "Your judgment for the complete response",
        options=list(reverse_display),
        horizontal=True,
        key=label_key,
        on_change=callback,
        args=callback_args,
    )
    st.caption(
        "Supported: material factual claims are correct. Hallucinated: at least "
        "one material claim is false or misleading. Abstain: uncertain or not "
        "enough information to judge. Do not mark a true claim hallucinated "
        "merely because it is not stated in the question."
    )
    st.text_area(
        "Notes (optional)",
        key=notes_key,
        on_change=callback,
        args=callback_args,
    )
    st.caption(f"Answer ID: {example_id} · autosaves to `{output_path}`")

    if st.session_state.get("last_saved_answer") == example_id:
        st.success("Annotation saved.")

    csv_data = annotation_csv(answers, annotations)
    st.download_button(
        "Download human_labels.csv",
        data=csv_data,
        file_name=Path(output_path).name or "human_labels.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
