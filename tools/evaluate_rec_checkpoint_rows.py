import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def normalize_digits(value):
    if value is None:
        return ""

    value = str(value).translate(DIGIT_TRANSLATION)
    value = re.sub(r"\D+", "", value)
    return value


def resolve_from_base(path_value, base_dir):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def to_cli_path(path):
    return str(path).replace("\\", "/")


def parse_infer_output(text):
    raw_pred = ""
    score = None

    for line in text.splitlines():
        if "result:" not in line:
            continue

        payload = line.split("result:", 1)[1].strip()
        parts = payload.split()

        if not parts:
            return "", None

        # Usually: result: 74900000    0.7946616411209106
        possible_score = parts[-1]
        try:
            score = float(possible_score)
            raw_pred = " ".join(parts[:-1]).strip()
        except ValueError:
            raw_pred = payload.strip()
            score = None

        return raw_pred, score

    return raw_pred, score


def extract_row_info(image_rel_path):
    stem = Path(image_rel_path).stem

    # Example: nek_001_row_21
    match = re.match(r"(.+)_row_(\d+)$", stem)
    if not match:
        return stem, None

    source_image_id = match.group(1)
    row_number = int(match.group(2))
    return source_image_id, row_number


def run_infer(python_exe, paddleocr_dir, config_path, checkpoint_path, image_path):
    cmd = [
        str(python_exe),
        "tools/infer_rec.py",
        "-c",
        to_cli_path(config_path),
        "-o",
        f"Global.checkpoints={to_cli_path(checkpoint_path)}",
        f"Global.infer_img={to_cli_path(image_path)}",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        cmd,
        cwd=str(paddleocr_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    raw_pred, score = parse_infer_output(combined_output)

    return {
        "returncode": completed.returncode,
        "raw_output": combined_output,
        "raw_pred": raw_pred,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paddleocr-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", default="test.txt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    project_dir = Path.cwd().resolve()

    paddleocr_dir = resolve_from_base(args.paddleocr_dir, project_dir)
    config_path = resolve_from_base(args.config, paddleocr_dir)
    dataset_dir = resolve_from_base(args.dataset, project_dir)
    checkpoint_path = resolve_from_base(args.checkpoint, project_dir)

    labels_path = Path(args.labels)
    if labels_path.is_absolute():
        labels_path = labels_path
    else:
        labels_path = dataset_dir / labels_path

    out_json = resolve_from_base(args.out_json, project_dir)
    out_csv = resolve_from_base(args.out_csv, project_dir)

    python_exe = Path(sys.executable).resolve()

    if not paddleocr_dir.exists():
        raise FileNotFoundError(f"PaddleOCR dir not found: {paddleocr_dir}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if not Path(str(checkpoint_path) + ".pdparams").exists():
        raise FileNotFoundError(f"Checkpoint .pdparams not found: {checkpoint_path}.pdparams")

    rows = []

    for line in labels_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid label line: {line}")

        image_rel, expected = parts
        rows.append({
            "image_rel": image_rel.strip(),
            "expected": expected.strip(),
        })

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    results = []

    for index, item in enumerate(rows, start=1):
        image_rel = item["image_rel"]
        expected = item["expected"]

        image_path = dataset_dir / image_rel.replace("/", os.sep)
        source_image_id, row_number = extract_row_info(image_rel)

        infer_result = run_infer(
            python_exe=python_exe,
            paddleocr_dir=paddleocr_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            image_path=image_path,
        )

        pred_raw = infer_result["raw_pred"]
        pred_norm = normalize_digits(pred_raw)
        expected_norm = normalize_digits(expected)

        correct = pred_norm == expected_norm
        missing = pred_norm == ""

        row_result = {
            "index": index,
            "source_image_id": source_image_id,
            "row": row_number,
            "image": image_rel,
            "expected": expected,
            "expected_norm": expected_norm,
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
            "score": infer_result["score"],
            "correct": correct,
            "missing": missing,
            "returncode": infer_result["returncode"],
        }

        results.append(row_result)

        status = "OK" if correct else "WRONG"
        print(
            f"[{index:03d}/{len(rows):03d}] {status} | "
            f"{image_rel} | expected={expected_norm} | pred={pred_norm} | score={infer_result['score']}"
        )

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    missing_count = sum(1 for r in results if r["missing"])
    wrong_count = total - correct_count

    summary = {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "missing": missing_count,
        "accuracy_percent": round((correct_count / total * 100) if total else 0, 2),
        "labels_path": str(labels_path),
        "checkpoint": str(checkpoint_path),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "source_image_id",
                "row",
                "image",
                "expected",
                "expected_norm",
                "pred_raw",
                "pred_norm",
                "score",
                "correct",
                "missing",
                "returncode",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("")
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("")
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()