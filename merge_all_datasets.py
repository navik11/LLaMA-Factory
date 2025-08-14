import json
import os
from typing import Dict, List

def merge_datasets(dataset_files: List[str], output_file: str) -> int:
    """
    Merge multiple ShareGPT JSON datasets into a single JSON file.
    Automatically handles missing files and ensures consistent format.
    """
    merged_data = []

    for ds_file in dataset_files:
        if not os.path.exists(ds_file):
            print(f"Skipping missing file: {ds_file}")
            continue

        with open(ds_file, "r", encoding="utf-8-sig") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in {ds_file}")

            if isinstance(data, dict):  # single entry, wrap in list
                data = [data]
            elif not isinstance(data, list):
                raise ValueError(f"Unexpected format in {ds_file}")

            merged_data.extend(data)
            print(f"Loaded {len(data)} entries from {ds_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully merged {len(merged_data)} entries.")
    print(f"Saved merged dataset to: {output_file}")
    return len(merged_data)


def create_and_update_dataset_info(dataset_name: str, data_file: str, dataset_info_path: str):
    """
    Creates or updates the dataset_info.json file for LLaMA-Factory ShareGPT datasets.
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
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            try:
                existing_info: Dict = json.load(f)
            except json.JSONDecodeError:
                existing_info = {}
    else:
        existing_info = {}

    existing_info.update(new_info)

    os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)

    print(f"📝 Updated dataset info at: {dataset_info_path}")


def main():
    # Input datasets
    dataset_dir = "data"
    dataset_files = [
        # os.path.join(dataset_dir, "crop_recommendation_sharegpt.json"),
        # os.path.join(dataset_dir, "qa_dataset_sharegpt.json"),
        os.path.join(dataset_dir, "plant_disease_sharegpt.json")
    ]

    # Output merged dataset
    merged_filename = "merged_dataset.json"
    merged_filepath = os.path.join(dataset_dir, merged_filename)
    merged_dataset_name = "merged_dataset"

    # Merge datasets
    total_entries = merge_datasets(dataset_files, merged_filepath)

    # Update dataset_info.json if merge was successful
    if total_entries > 0:
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
        create_and_update_dataset_info(merged_dataset_name, merged_filename, dataset_info_path)

    print("\n--- Merge Complete! ---")
    print(f"✓ Output file: {merged_filename}")
    print(f"✓ Dataset name for your YAML config: {merged_dataset_name}")
    print("You can now run LLaMA-Factory with LLaVA-NeXT using the merged dataset.")


if __name__ == "__main__":
    main()
