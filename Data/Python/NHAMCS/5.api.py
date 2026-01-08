import json
import requests
import time
import os
from typing import List, Dict, Optional


def read_json_data(filename: str) -> List[Dict]:
    """Read a JSON file (list of records or a single dict)."""
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def is_valid_triage(triage_value: str) -> bool:
    """Check whether triage is one of '1'..'5'."""
    return bool(triage_value) and triage_value in {"1", "2", "3", "4", "5"}


def should_process_record(triage_value: str, triage_counters: Dict[str, int]) -> bool:
    """Decide whether to process based on triage quotas."""
    triage_limits = {
        "1": float("inf"),
        "2": 900,
        "3": 1000,
        "4": 900,
        "5": 400,
    }
    current_count = triage_counters.get(triage_value, 0)
    limit = triage_limits.get(triage_value, 0)
    return current_count < limit


def create_polish_prompt(template_txt: str, match_text1: str, match_text2: str, match_text3: str) -> str:
    """Build a prompt for polishing the template text."""
    return f"""Please polish and improve the following emergency department patient description to make it more natural, fluent, and clinically appropriate, similar to the writing style of the examples provided.

EXAMPLES:
1. "{match_text1}"
2. "{match_text2}"
3. "{match_text3}"

ORIGINAL TEXT TO POLISH:
"{template_txt}"

Please rewrite and expand the original text using a more natural and professional medical narrative style while retaining all key clinical information. Output only the polished version without any additional explanations."""


def call_deepseek_api(api_key: str, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call DeepSeek API with retry."""
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print(f"  Request timeout, retrying {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"  API call error: {e}, retrying {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(2)

        except KeyError as e:
            print(f"  Response parsing error: {e}")
            return None

        except Exception as e:
            print(f"  Unknown error: {e}")
            return None

    print("  All retry attempts failed")
    return None


def load_existing_results(filename: str) -> List[Dict]:
    """Load existing output JSON if present."""
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return []
    return []


def save_single_result(result: Dict, filename: str):
    """Append one record to the output JSON (rewrite the full file)."""
    existing_results = load_existing_results(filename)
    existing_results.append(result)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(existing_results, file, indent=2, ensure_ascii=False)

    print(f"  Result saved to: {filename}")


def main():
    # Configuration (placeholders for GitHub)
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
    INPUT_JSON_PATH = "/path/to/your/INPUT.json"
    OUTPUT_JSON_PATH = "/path/to/your/OUTPUT.json"

    if not API_KEY or API_KEY == "YOUR_DEEPSEEK_API_KEY":
        print("Error: Please set DEEPSEEK_API_KEY env var (recommended) or replace YOUR_DEEPSEEK_API_KEY.")
        return

    try:
        print("Reading JSON file...")
        all_data = read_json_data(INPUT_JSON_PATH)
        print(f"Total records read: {len(all_data)}")

        existing_results = load_existing_results(OUTPUT_JSON_PATH)
        processed_ids = {r.get("id") for r in existing_results if "id" in r}

        skipped_records = []
        failed_records = []
        processed_records = len(existing_results)

        triage_counters = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for r in existing_results:
            triage = r.get("triage", "")
            if triage in triage_counters:
                triage_counters[triage] += 1

        print(f"Found {processed_records} existing results")

        for i, record in enumerate(all_data, 1):
            record_id = record.get("id", "N/A")

            if record_id in processed_ids:
                print(f"Skipping already processed record {i}/{len(all_data)} (ID: {record_id})...")
                continue

            print(f"Processing record {i}/{len(all_data)} (ID: {record_id})...")

            try:
                triage_value = record.get("Triage", "")
                if not is_valid_triage(triage_value):
                    print(f"  Skipped: Invalid triage value '{triage_value}'")
                    skipped_records.append(record_id)
                    continue

                if not should_process_record(triage_value, triage_counters):
                    print(f"  Skipped: Reached limit for triage level {triage_value}")
                    skipped_records.append(record_id)
                    continue

                template_txt = record.get("template_txt", "")
                match_text1 = record.get("match_text1", "")
                match_text2 = record.get("match_text2", "")
                match_text3 = record.get("match_text3", "")

                if not all([template_txt, match_text1, match_text2, match_text3]):
                    print("  Skipped: Missing required fields")
                    skipped_records.append(record_id)
                    continue

                prompt = create_polish_prompt(template_txt, match_text1, match_text2, match_text3)

                print("  Calling API to polish text...")
                answer = call_deepseek_api(API_KEY, prompt)

                if answer:
                    result_record = {
                        "id": f"2020_{record_id}",
                        "answer": answer.strip('"').strip(),
                        "triage": triage_value,
                    }

                    save_single_result(result_record, OUTPUT_JSON_PATH)

                    triage_counters[triage_value] += 1
                    processed_records += 1
                    processed_ids.add(record_id)

                    print(
                        f"  Record {i} polished and saved successfully "
                        f"(Triage {triage_value}: {triage_counters[triage_value]})"
                    )
                else:
                    failed_records.append(record_id)
                    print(f"  Record {i} polishing failed")

                if i < len(all_data):
                    time.sleep(1)

            except Exception as e:
                print(f"  Error processing record {i}: {e}")
                failed_records.append(record_id)

        print("\nProcessing completed!")
        print(f"Total records: {len(all_data)}")
        print(f"Successfully processed: {processed_records}")
        print(f"Skipped records: {len(skipped_records)}")
        print(f"Failed records: {len(failed_records)}")

        print("\nTriage level processing summary:")
        limits_display = {"1": "unlimited", "2": "900", "3": "1000", "4": "900", "5": "400"}
        for triage in ["1", "2", "3", "4", "5"]:
            print(f"  Triage {triage}: {triage_counters[triage]} processed (limit: {limits_display[triage]})")

        if skipped_records:
            head = skipped_records[:10]
            print(f"\nSkipped record IDs (first 10): {head}{'...' if len(skipped_records) > 10 else ''}")

        if failed_records:
            head = failed_records[:10]
            print(f"Failed record IDs (first 10): {head}{'...' if len(failed_records) > 10 else ''}")

    except FileNotFoundError:
        print(f"Error: File not found {INPUT_JSON_PATH}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {INPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
