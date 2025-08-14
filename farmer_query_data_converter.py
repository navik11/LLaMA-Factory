import csv
import json
import os
from typing import Dict

def convert_qa_csv_to_llamafactory(input_csv: str, output_file: str):
    """
    Convert Q&A CSV dataset to LLaMA-Factory ShareGPT format (text-only).
    Automatically cleans header names and handles CSV encoding issues.
    """
    converted_data = []

    with open(input_csv, 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM if present
        reader = csv.DictReader(f)
        # Clean up fieldnames
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        required_cols = ["questions", "answers"]
        missing = [col for col in required_cols if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        for row in reader:
            # Strip spaces from each field value
            row = {k.strip(): v.strip() for k, v in row.items()}
            question = row['questions']
            answer = row['answers']

            converted_data.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer}
                ],
                "images": []  # No images for this dataset
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully converted {len(converted_data)} Q&A entries.")
    print(f"Saved to: {output_file}")
    return len(converted_data)


def create_and_update_dataset_info(dataset_name: str, data_file: str, dataset_info_path: str):
    """
    Creates the correct dataset_info.json entry and updates the file.
    """
    new_info = {
        dataset_name: {
            "file_name": data_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            }
        }
    }

    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            try:
                existing_info: Dict = json.load(f)
            except json.JSONDecodeError:
                existing_info = {}
    else:
        existing_info = {}

    existing_info.update(new_info)

    os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)

    print(f"Updated dataset info at: {dataset_info_path}")


def main():
    output_folder = "data"
    qa_csv_input = "farmer_call_query_dataset_dropped.csv"
    qa_output_filename = "qa_dataset_sharegpt.json"
    qa_dataset_name = "qa_text"

    if os.path.exists(qa_csv_input):
        qa_output_filepath = os.path.join(output_folder, qa_output_filename)
        num_qa_entries = convert_qa_csv_to_llamafactory(qa_csv_input, qa_output_filepath)

        if num_qa_entries > 0:
            dataset_info_path = os.path.join(output_folder, "dataset_info.json")
            create_and_update_dataset_info(qa_dataset_name, qa_output_filename, dataset_info_path)

    print("\n--- Conversion Complete! ---")
    print(f"✓ Output file: {qa_output_filename}")
    print(f"✓ Dataset name for your YAML config: {qa_dataset_name}")
    print("You can now run LLaMA-Factory with LLaVA-NeXT using the updated configuration.")


if __name__ == "__main__":
    main()
