#!/usr/bin/env python3
"""
Convert CDDM_converted.jsonl to the LLaMA-Factory ShareGPT format for LLaVA-NeXT.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

def convert_conversation(messages: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Converts a single conversation from the input format to the target LLaMA-Factory format.

    The target format requires:
    1. A top-level "images" key containing a list of image paths.
    2. A "<image>" placeholder token in the first user message's content.
    """
    llama_factory_turns = []
    image_paths = []
    is_first_user_message = True

    for message in messages:
        role = message['role']
        content = message['content'].strip()

        # Determine the speaker role for ShareGPT format
        if role == 'user':
            speaker = 'human'
        elif role == 'assistant':
            speaker = 'gpt'
        else:
            # Skip unknown roles
            continue

        # Process the first user message to extract the image and add the placeholder
        if speaker == 'human' and is_first_user_message:
            img_pattern = r'<img>(.*?)</img>'
            img_match = re.search(img_pattern, content)

            if img_match:
                # 1. Extract the relative image path and add it to our list.
                relative_path = f"/kaggle/input/cddm-dataset/dataset{img_match.group(1)}"
                image_paths.append(relative_path)

                # 2. Remove the old <img> tag and add the required <image> placeholder.
                text_content = re.sub(img_pattern, '', content).strip()
                # For LLaVA, the image token should be at the beginning
                if text_content:
                    value = f"<image>\n{text_content}"
                else:
                    value = "<image>"
            else:
                # If the first user message has no image, we can't use it for V-L training.
                # We will skip this entire conversation entry.
                return None
            
            # Ensure we only process the image for the very first user turn
            is_first_user_message = False
        else:
            # For assistant messages or subsequent user messages, just use the content as is.
            value = content

        llama_factory_turns.append({"from": speaker, "value": value})
    
    # If for some reason we have turns but no image was extracted, skip.
    if not image_paths:
        return None

    # Construct the final object in the correct format
    final_conversation = {
        "conversations": llama_factory_turns,
        "images": image_paths  # Plural key with a list of paths
    }

    return final_conversation


def convert_jsonl_to_llamafactory(input_file: str, output_file: str):
    """
    Reads the input JSONL file and writes the converted data to a new JSON file.
    """
    converted_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                messages = data.get('messages')
                if not messages:
                    continue

                # Convert the conversation using our new logic
                converted_entry = convert_conversation(messages)
                
                # Only add if the conversion was successful (i.e., returned an object)
                if converted_entry:
                    converted_data.append(converted_entry)

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {line_num + 1}.")
                continue
    
    # Save the final list of converted data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # The output is a single JSON array, not JSONL
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully converted {len(converted_data)} conversations.")
    print(f"Saved to: {output_file}")
    return len(converted_data)

def create_and_update_dataset_info(dataset_name: str, data_file: str, dataset_info_path: str):
    """
    Creates the correct dataset_info.json entry and updates the file.
    """
    # This structure now matches the LLaMA-Factory documentation for LLaVA-NeXT
    new_info = {
        dataset_name: {
            "file_name": data_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"  # Corrected to plural "images"
            }
        }
    }
    
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            try:
                existing_info = json.load(f)
            except json.JSONDecodeError:
                existing_info = {} # Start fresh if file is corrupt
    else:
        existing_info = {}
    
    existing_info.update(new_info)
    
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)
    
    print(f"Updated dataset info at: {dataset_info_path}")

def main():
    # --- Configuration ---
    # NOTE: It's assumed you run this from the root of your project,
    # and LLaMA-Factory is a subdirectory.
    input_file = "CDDM_converted.jsonl" # Expects this file in the same directory
    output_folder = "data" # We will save the output here
    output_filename = "plant_disease_sharegpt.json"
    dataset_name = "plant_disease_vlm"
    
    # --- Execution ---
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please place the script in your project's root directory and ensure the data file is present.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(output_folder, output_filename)
    
    print("Starting conversion to LLaMA-Factory ShareGPT format for LLaVA-NeXT...")
    num_conversations = convert_jsonl_to_llamafactory(input_file, output_filepath)
    
    if num_conversations > 0:
        # Define path to LLaMA-Factory's dataset info file
        dataset_info_path = os.path.join("data", "dataset_info.json")
        if not os.path.exists(os.path.dirname(dataset_info_path)):
             print(f"Warning: 'data' directory not found. Cannot update dataset_info.json.")
        else:
             create_and_update_dataset_info(dataset_name, output_filename, dataset_info_path)
    
    print("\n--- Conversion Complete! ---")
    print(f"✓ Output file: {output_filepath}")
    print(f"✓ Dataset name for your YAML config: {dataset_name}")
    print("You can now run LLaMA-Factory with LLaVA-NeXT using the updated configuration.")

if __name__ == "__main__":
    main()