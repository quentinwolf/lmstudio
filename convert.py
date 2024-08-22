import json
import os
import unicodedata

def unicode_to_ascii(text):
    if isinstance(text, list):
        return [unicode_to_ascii(item) for item in text]
    elif isinstance(text, str):
        return ''.join(
            char if ord(char) < 128 else
            "'" if char == u'\u2019' else  # Unicode apostrophe
            char
            for char in unicodedata.normalize('NFKD', text)
        )
    else:
        return text

def safe_unicode_to_ascii(content):
    if isinstance(content, dict):
        return {k: safe_unicode_to_ascii(v) for k, v in content.items()}
    elif isinstance(content, list):
        return [safe_unicode_to_ascii(item) for item in content]
    else:
        return unicode_to_ascii(content)

def convert_path_to_relative(path):
    # Split the path and take the last three components
    components = path.replace('\\', '/').split('/')
    return '/'.join(components[-3:])

def convert_chat(chat_file, config_file, metadata_file):
    try:
        # Load the old format files with UTF-8 encoding
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_data = json.load(f)

        # Convert the model path to relative format
        relative_model_path = convert_path_to_relative(metadata_data["lastUsedModel"]["filePath"])

        # Create the new format structure
        new_format = {
            "name": safe_unicode_to_ascii(metadata_data.get("name", "Converted Chat")),
            "systemPrompt": safe_unicode_to_ascii(config_data["inference_params"].get("pre_prompt", "")),
            "messages": [],
            "usePerChatPredictionConfig": True,
            "perChatPredictionConfig": {"fields": []},
            "tokenCount": metadata_data["stats"]["predictionStats"]["tokenCount"],
            "createdAt": int(metadata_data["identifier"]),
            "pinned": False,
            "clientInput": "",
            "clientInputFiles": [],
            "userFilesSizeBytes": 0,
            "lastUsedModel": {
                "indexedModelIdentifier": relative_model_path,
                "identifier": safe_unicode_to_ascii(os.path.splitext(os.path.basename(metadata_data["lastUsedModel"]["filePath"]))[0].lower()),
                "instanceLoadTimeConfig": {"fields": []},
                "instanceOperationTimeConfig": {"fields": []}
            },
            "notes": []
        }

        # Prepare genInfo
        gen_info = {
            "indexedModelIdentifier": relative_model_path,
            "identifier": safe_unicode_to_ascii(os.path.splitext(os.path.basename(metadata_data["lastUsedModel"]["filePath"]))[0].lower()),
            "loadModelConfig": {
                "fields": [
                    {"key": f"llm.load.{k}", "value": v} for k, v in config_data["load_params"].items()
                ]
            },
            "predictionConfig": {
                "fields": [
                    {"key": f"llm.prediction.{k}", "value": v} for k, v in config_data["inference_params"].items() if k != "pre_prompt"
                ]
            },
            "stats": metadata_data["stats"]
        }

        # Convert messages
        for msg in chat_data["messages"]:
            new_msg = {
                "versions": [{
                    "type": "singleStep" if msg["role"] == "user" else "multiStep",
                    "role": msg["role"],
                    "content": [{"type": "text", "text": safe_unicode_to_ascii(msg["content"])}],
                    "processed": [{
                        "role": msg["role"],
                        "content": [{"type": "text", "text": safe_unicode_to_ascii(msg["content"])}]
                    }]
                }],
                "currentlySelected": 0
            }
            if msg["role"] == "assistant":
                new_msg["versions"][0]["steps"] = [
                    {
                        "type": "contentBlock",
                        "stepIdentifier": f"{metadata_data['identifier']}-{msg['content'][:20]}",
                        "content": [{"type": "text", "text": safe_unicode_to_ascii(msg["content"])}],
                        "defaultShouldIncludeInContext": True,
                        "shouldIncludeInContext": True,
                        "genInfo": gen_info
                    }
                ]
            new_format["messages"].append(new_msg)

        # Add configuration
        new_format["usePerChatPredictionConfig"] = True
        new_format["perChatPredictionConfig"] = {
            "fields": [
                {"key": "llm.prediction.systemPrompt", "value": safe_unicode_to_ascii(config_data["inference_params"].get("pre_prompt", ""))}
            ] + [
                {"key": f"llm.load.{k}", "value": v} for k, v in config_data["load_params"].items()
            ] + [
                {"key": f"llm.prediction.{k}", "value": v} for k, v in config_data["inference_params"].items() if k != "pre_prompt"
            ]
        }

        return new_format, None
    except Exception as e:
        return None, str(e)
        

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".chat.json"):
            base_name = filename.split('.')[0]
            chat_file = os.path.join(input_dir, f"{base_name}.chat.json")
            config_file = os.path.join(input_dir, f"{base_name}.config.chat.json")
            metadata_file = os.path.join(input_dir, f"{base_name}.metadata.chat.json")

            if os.path.exists(chat_file) and os.path.exists(config_file) and os.path.exists(metadata_file):
                new_format, error = convert_chat(chat_file, config_file, metadata_file)
                if error:
                    print(f"Error converting {base_name}: {error}")
                else:
                    output_file = os.path.join(output_dir, f"{base_name}.conversation.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(new_format, f, indent=2, ensure_ascii=False)
                    print(f"Successfully converted {base_name} to new format.")
            else:
                print(f"Skipping {base_name} due to missing files.")

if __name__ == "__main__":
    input_directory = input("Enter the directory containing old LM Studio chat files: ")
    output_directory = input("Enter the directory to save converted files: ")
    process_directory(input_directory, output_directory)
    print("Conversion process completed. Check the output for any error messages.")
