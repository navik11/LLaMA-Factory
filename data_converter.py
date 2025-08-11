#!/usr/bin/env python3
"""
Convert CDDM_converted.jsonl to LLaMA-Factory format for Qwen-VL training
"""

import json
import os
import re
from typing import List, Dict, Any

def convert_jsonl_to_llamafactory(input_file: str, output_file: str, dataset_name: str = "plant_disease"):
    """
    Convert JSONL format to LLaMA-Factory format for vision-language training
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    messages = data['messages']
                    
                    # Convert conversation to LLaMA-Factory format
                    converted_conversation = convert_conversation(messages, line_num)
                    if converted_conversation:
                        converted_data.append(converted_conversation)
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                    continue
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(converted_data)} conversations")
    print(f"Saved to: {output_file}")
    
    return len(converted_data)

def convert_conversation(messages: List[Dict], conversation_id: int) -> Dict[str, Any]:
    """
    Convert a single conversation to LLaMA-Factory format
    """
    conversation = {
        "conversation_id": conversation_id,
        "conversations": []
    }
    
    image_path = None
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            # Extract image path if present
            img_pattern = r'<img>(.*?)</img>'
            img_match = re.search(img_pattern, content)
            
            if img_match:
                image_path = img_match.group(1)
                # Remove image tag from content
                text_content = re.sub(img_pattern, '', content).strip()
                
                # For LLaMA-Factory, we include image in the user message
                conversation["conversations"].append({
                    "from": "human",
                    "value": text_content,
                    "image": f"/kaggle/input/cddm-dataset/dataset{image_path}"  # LLaMA-Factory format for images
                })
            else:
                conversation["conversations"].append({
                    "from": "human", 
                    "value": content
                })
                
        elif role == 'assistant':
            conversation["conversations"].append({
                "from": "gpt",
                "value": content
            })
    
    # Add image path to conversation metadata if found
    if image_path:
        conversation["image"] = image_path
    
    return conversation

def create_dataset_info(dataset_name: str, data_file: str) -> Dict[str, Any]:
    """
    Create dataset_info.json entry for LLaMA-Factory
    """
    dataset_info = {
        dataset_name: {
            "file_name": data_file,
            "formatting": "sharegpt",  # LLaMA-Factory format for conversations
            "columns": {
                "messages": "conversations",
                "images": "image"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value", 
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    }
    return dataset_info

def update_dataset_info_file(dataset_name: str, data_file: str, dataset_info_path: str):
    """
    Update or create dataset_info.json file
    """
    new_info = create_dataset_info(dataset_name, data_file)
    
    # Load existing dataset_info.json if it exists
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            existing_info = json.load(f)
        existing_info.update(new_info)
    else:
        existing_info = new_info
    
    # Save updated dataset_info.json
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)
    
    print(f"Updated dataset_info.json with {dataset_name}")

def main():
    # Configuration
    input_file = "../kaggle/input/cddm-dataset/CDDM_converted.jsonl"
    output_file = "data/plant_disease_data.json"
    dataset_name = "plant_disease"
    dataset_info_path = "data/dataset_info.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Make sure CDDM_converted.jsonl is in the current directory.")
        return
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Convert data
    print("Converting data to LLaMA-Factory format...")
    num_conversations = convert_jsonl_to_llamafactory(input_file, output_file, dataset_name)
    
    # Update dataset_info.json
    print("Updating dataset_info.json...")
    update_dataset_info_file(dataset_name, "plant_disease_data.json", dataset_info_path)
    
    print(f"\nConversion completed!")
    print(f"✓ Converted {num_conversations} conversations")
    print(f"✓ Data saved to: {output_file}")
    print(f"✓ Dataset info updated: {dataset_info_path}")
    print("\nYou can now proceed with training using LLaMA-Factory!")

if __name__ == "__main__":
    main()