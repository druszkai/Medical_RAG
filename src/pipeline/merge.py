import json

from src.config import *


def merge_json_files(input_files, output_file):
    merged_data = []

    print("Fájlok összevonása indult...\n")

    for file_name in input_files:
        file_path = PROCESSED_DATA_DIR / file_name
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged_data.extend(data)
            print(f"[{file_name}] hozzáadva: {len(data)} dokumentum.")
        except FileNotFoundError:
            print(f"Hiba: A {file_path} nem található, kihagyva.")
        except json.JSONDecodeError:
            print(f"Hiba: A {file_path} nem érvényes JSON formátumú, kihagyva.")

    output_path = PROCESSED_DATA_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"\nSiker! Összesen {len(merged_data)} dokumentum összevonva és mentve ide: {output_file}")


if __name__ == "__main__":
    files_to_merge = [
        "cleaned_data_en_hu.json",
        "pubmed_1000_en.json",
        "webmd_formatted_articles.json",
        "scraped_articles.json",
    ]
    output_filename = "all_merged_articles.json"

    merge_json_files(files_to_merge, output_filename)