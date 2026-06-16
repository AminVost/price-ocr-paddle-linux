import json
from pathlib import Path
from paddleocr import TextRecognition


PROJECT_DIR = Path(__file__).resolve().parents[1]

CROPS_ROOT = PROJECT_DIR / "experiments/product_text_ocr/crops_v2"
OUTPUT_DIR = PROJECT_DIR / "experiments/product_text_ocr/output"
OUTPUT_JSON = OUTPUT_DIR / "arabic_rec_variants_results.json"
OUTPUT_TXT = OUTPUT_DIR / "arabic_rec_variants_results.txt"

MODEL_NAME = "arabic_PP-OCRv5_mobile_rec"

VARIANTS = [
    "raw",
    "gray_up4",
    "sharp_up4",
    "bw_up4",
]


def extract_result(res):
    """
    PaddleX result object can be printed, but for stable saving
    we try to read its json-like internal data if available.
    """
    data = None

    if hasattr(res, "json"):
        try:
            data = res.json
        except Exception:
            data = None

    if isinstance(data, dict):
        inner = data.get("res", data)
        return {
            "rec_text": inner.get("rec_text", ""),
            "rec_score": inner.get("rec_score", None),
            "raw": inner,
        }

    if hasattr(res, "__dict__"):
        d = getattr(res, "__dict__", {})
        return {
            "rec_text": d.get("rec_text", ""),
            "rec_score": d.get("rec_score", None),
            "raw": str(res),
        }

    return {
        "rec_text": "",
        "rec_score": None,
        "raw": str(res),
    }


def main():
    if not CROPS_ROOT.exists():
        raise FileNotFoundError(f"Crops root not found: {CROPS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = TextRecognition(model_name=MODEL_NAME)

    all_results = []

    for variant in VARIANTS:
        variant_dir = CROPS_ROOT / variant

        if not variant_dir.exists():
            print(f"Skipping missing variant: {variant_dir}")
            continue

        image_paths = sorted(variant_dir.glob("nek_product_row_*.png"))

        print("")
        print("=" * 100)
        print(f"VARIANT: {variant}")
        print(f"Images: {len(image_paths)}")
        print("=" * 100)

        for image_path in image_paths:
            row_num = int(image_path.stem.split("_")[-1])

            output = model.predict(input=str(image_path), batch_size=1)

            # Usually one result for one image
            parsed_items = []

            for res in output:
                parsed = extract_result(res)
                parsed_items.append(parsed)

            if parsed_items:
                rec_text = parsed_items[0]["rec_text"]
                rec_score = parsed_items[0]["rec_score"]
            else:
                rec_text = ""
                rec_score = None

            item = {
                "variant": variant,
                "row": row_num,
                "image": str(image_path).replace("\\", "/"),
                "rec_text": rec_text,
                "rec_score": rec_score,
            }

            all_results.append(item)

            print(
                f"row {row_num:02d} | "
                f"score={rec_score} | "
                f"text={rec_text}"
            )

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "results": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = []
    current_variant = None

    for item in all_results:
        if item["variant"] != current_variant:
            current_variant = item["variant"]
            lines.append("")
            lines.append("=" * 100)
            lines.append(f"VARIANT: {current_variant}")
            lines.append("=" * 100)

        lines.append(
            f"row {item['row']:02d} | "
            f"score={item['rec_score']} | "
            f"{item['rec_text']}"
        )

    OUTPUT_TXT.write_text("\n".join(lines).strip(), encoding="utf-8")

    print("")
    print("Saved:")
    print(OUTPUT_JSON)
    print(OUTPUT_TXT)


if __name__ == "__main__":
    main()