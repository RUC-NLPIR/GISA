import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval_script.run_evaluation import SimpleEvaluator

TABLE_CSV = "City,Population\nAlpha,120\nBeta,180\n"
TABLE_ROWS = "Alpha\t120\nBeta\t180"
SCORE_NAMES = ("precision", "recall", "f1")
FULL = (1.0, 1.0, 1.0)
ZERO = (0.0, 0.0, 0.0)
HALF = (0.5, 0.5, 0.5)


def fenced_tsv(content):
    return f"Here is the table:\n```tsv\n{content}\n```"


@pytest.mark.parametrize(
    ("prediction", "em", "row_scores", "item_scores"),
    [
        pytest.param("City\tPopulation\n" + TABLE_ROWS, 1, FULL, FULL, id="correct"),
        pytest.param(
            "City\tElevation\n" + TABLE_ROWS, 0, FULL, HALF, id="wrong-header"
        ),
        pytest.param(
            "District\tElevation\n" + TABLE_ROWS, 0, ZERO, ZERO, id="all-renamed"
        ),
        pytest.param(
            "Population\tCity\n" + TABLE_ROWS, 0, ZERO, ZERO, id="swapped-labels"
        ),
        pytest.param(
            "Population\tCity\n120\tAlpha\n180\tBeta",
            0,
            FULL,
            FULL,
            id="reordered-columns",
        ),
        pytest.param(
            "City\tPopulation\nAlpha\t120\nBeta\t181",
            0,
            HALF,
            (0.75, 0.75, 0.75),
            id="wrong-cells",
        ),
        pytest.param(
            " CITY \t Popu LATION \nAlpha\t$120.00\nBeta\t18,000%",
            1,
            FULL,
            FULL,
            id="normalized-headers-and-numbers",
        ),
        pytest.param(
            "City\nAlpha\nBeta", 0, FULL, (1.0, 0.5, 2 / 3), id="missing-column"
        ),
    ],
)
def test_table_matches_headers_and_cells(
    tmp_path, prediction, em, row_scores, item_scores
):
    gold = tmp_path / "gold.csv"
    gold.write_text(TABLE_CSV, encoding="utf-8")

    result = SimpleEvaluator().evaluate_one(fenced_tsv(prediction), str(gold), "table")

    for kind, expected in (("row", row_scores), ("item", item_scores)):
        assert tuple(
            result[f"table_{kind}_{name}"] for name in SCORE_NAMES
        ) == pytest.approx(expected)
    assert result["question_type"] == "table"
    assert type(result["global_em"]) is int
    assert result["global_em"] == em


@pytest.mark.parametrize(
    ("gold_text", "prediction", "em"),
    [
        ("Left,Right\n1,1\n2,2\n", "Right\tLeft\n1\t1\n2\t2", 0),
        ("Label,Label\nA,B\nC,D\n", "Label\tLabel\nA\tB\nC\tD", 1),
        ("Label,Label\nA,B\nC,D\n", "Label\tOther\nA\tB\nC\tD", 0),
        ("City\nAlpha\nBeta\n", "CITY\nAlpha\nBeta", 1),
        ("City\nAlpha\nBeta\n", "Place\nAlpha\nBeta", 0),
    ],
)
def test_header_structure_uses_pandas_parsed_columns(
    tmp_path, gold_text, prediction, em
):
    gold = tmp_path / "gold.csv"
    gold.write_text(gold_text, encoding="utf-8")

    result = SimpleEvaluator().evaluate_one(fenced_tsv(prediction), str(gold), "table")

    assert result["global_em"] == em


@pytest.mark.parametrize("header", ["City,Population", "City,Elevation"])
def test_csv_prediction_has_same_semantics_as_tsv(tmp_path, header):
    gold = tmp_path / "gold.csv"
    gold.write_text(TABLE_CSV, encoding="utf-8")
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(f"{header}\nAlpha,120\nBeta,180\n", encoding="utf-8")
    original = prediction.read_bytes()
    evaluator = SimpleEvaluator()

    csv_result = evaluator.evaluate_one(str(prediction), str(gold), "table")
    tsv = fenced_tsv(header.replace(",", "\t") + "\n" + TABLE_ROWS)
    tsv_result = evaluator.evaluate_one(tsv, str(gold), "table")

    assert prediction.read_bytes() == original
    assert csv_result == tsv_result
    assert csv_result["global_em"] == (1 if header == "City,Population" else 0)


@pytest.mark.parametrize(
    ("answer_type", "gold_text", "prediction", "expected"),
    [
        ("item", "42\n", "Value\n42", {"item_em": 1, "global_em": 1}),
        (
            "list",
            "A\nB\n",
            "Item\nA\nB",
            {"list_content_f1": 1.0, "list_order_score": 1.0, "global_em": 1},
        ),
        (
            "list",
            "A\nB\n",
            "Item\nB\nA",
            {"list_content_f1": 1.0, "list_order_score": 0.5, "global_em": 0},
        ),
        (
            "set",
            "A\nB\n",
            "Item\nB\nA",
            {"set_precision": 1.0, "set_recall": 1.0, "set_f1": 1.0, "global_em": 1},
        ),
        (
            "set",
            "A\nB\n",
            "Item\nB\nC",
            {"set_precision": 0.5, "set_recall": 0.5, "set_f1": 0.5, "global_em": 0},
        ),
        ("item", "42\n", "", {"item_em": 0, "global_em": 0}),
    ],
)
def test_non_table_metrics_remain_compatible(
    tmp_path, answer_type, gold_text, prediction, expected
):
    gold = tmp_path / "gold.csv"
    gold.write_text(gold_text, encoding="utf-8")

    result = SimpleEvaluator().evaluate_one(prediction, str(gold), answer_type)

    assert result == {**expected, "question_type": answer_type}


def test_empty_table_prediction_retains_zero_scores(tmp_path):
    gold = tmp_path / "gold.csv"
    gold.write_text(TABLE_CSV, encoding="utf-8")

    result = SimpleEvaluator().evaluate_one("", str(gold), "table")

    assert result == {
        **{
            f"table_{kind}_{name}": 0.0
            for kind in ("row", "item")
            for name in SCORE_NAMES
        },
        "global_em": 0,
        "question_type": "table",
    }


def test_evaluation_cli_aggregates_exact_match_and_keeps_f1(tmp_path):
    pred_dir = tmp_path / "pred"
    gold_dir = tmp_path / "gold"
    pred_dir.mkdir()
    gold_dir.mkdir()
    questions = tmp_path / "questions.jsonl"
    # The existing CLI uses string IDs for JSON filenames and integer IDs for CSV filenames.
    samples = [("wrong", "table"), (2, "table"), ("item", "item")]
    questions.write_text(
        "".join(
            json.dumps({"id": qid, "answer_type": kind}) + "\n" for qid, kind in samples
        ),
        encoding="utf-8",
    )
    for qid, kind in samples:
        (gold_dir / f"{qid}.csv").write_text(
            TABLE_CSV if kind == "table" else "42\n", encoding="utf-8"
        )
    inputs = {
        pred_dir / "wrong.json": json.dumps(
            {"prediction": fenced_tsv("City\tElevation\n" + TABLE_ROWS)}
        ),
        pred_dir / "2.csv": TABLE_CSV,
        pred_dir / "item.json": json.dumps({"prediction": "Value\n42"}),
    }
    for path, contents in inputs.items():
        path.write_text(contents, encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "eval_script" / "run_evaluation.py"

    subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--pred_dir_path",
            str(pred_dir),
            "--gt_dir_path",
            str(gold_dir),
            "--question_file_path",
            str(questions),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    for path, contents in inputs.items():
        assert path.read_text(encoding="utf-8") == contents
    detailed = json.loads(
        (pred_dir / "_all_evaluation_results.json").read_text(encoding="utf-8")
    )
    scores = json.loads((pred_dir / "_final_scores.json").read_text(encoding="utf-8"))
    table = scores["table"]
    assert table["num_samples"] == 2
    for name in SCORE_NAMES:
        assert table[f"overall_table_row_{name}"] == 1.0
        assert table[f"overall_table_item_{name}"] == 0.75
    assert scores["item"] == {
        "num_samples": 1,
        "overall_item_em": 1.0,
        "overall_global_em": 1.0,
    }
    assert {key: row["metrics"]["global_em"] for key, row in detailed.items()} == {
        "wrong": 0,
        "2": 1,
        "item": 1,
    }
    assert table["overall_global_em"] == 0.5
    assert scores["overall_global_em"] == pytest.approx(2 / 3)
