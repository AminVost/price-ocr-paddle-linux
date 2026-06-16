from pathlib import Path
from paddleocr import TextRecognition

PROJECT_DIR = Path(__file__).resolve().parents[1]

CROPS_DIR = PROJECT_DIR / "experiments/product_text_ocr/crops"
OUTPUT_TXT = PROJECT_DIR / "experiments/product_text_ocr/output/arabic_rec_results.txt"
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

# Official PP-OCRv5 multilingual recognition model for Arabic/Persian/English family
model = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")

image_paths = sorted(CROPS_DIR.glob("nek_product_row_*.png"))

lines = []

for image_path in image_paths:
    print("=" * 80)
    print(f"IMAGE: {image_path.name}")

    try:
        results = model.predict(input=str(image_path), batch_size=1)

        result_texts = []

        for res in results:
            # Print full PaddleOCR result for debugging
            res.print()

            # Try to collect result as string safely
            result_texts.append(str(res))

        block = f"IMAGE: {image_path.name}\n" + "\n".join(result_texts)
        lines.append(block)

    except Exception as e:
        error_text = f"ERROR on {image_path.name}: {e}"
        print(error_text)
        lines.append(error_text)

OUTPUT_TXT.write_text("\n\n".join(lines), encoding="utf-8")
print("")
print(f"Saved result text to: {OUTPUT_TXT}")