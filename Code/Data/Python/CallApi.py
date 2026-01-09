import json
import requests
import time
import os
from typing import List, Dict, Optional


def read_json_data(filename: str) -> List[Dict]:
    """Read a JSON file."""
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    # If the data is a dict, convert it to a list
    if isinstance(data, dict):
        data = [data]

    return data


def is_valid_triage(triage_value: str) -> bool:
    """Check whether triage is a digit from 1 to 5."""
    if not triage_value:
        return False
    return triage_value in ["1", "2", "3", "4", "5"]


def should_process_record(triage_value: str, triage_counters: Dict[str, int]) -> bool:
    """Decide whether to process a record based on triage value and counters."""
    triage_limits = {
        "1": float("inf"),  # Level 1: unlimited
        "2": float("inf"),  # Level 2: unlimited
        "3": float("inf"),  # Level 3: unlimited
        "4": float("inf"),  # Level 4: unlimited
        "5": float("inf"),  # Level 5: unlimited
    }

    current_count = triage_counters.get(triage_value, 0)
    limit = triage_limits.get(triage_value, 0)

    return current_count < limit


def create_polish_prompt(template_txt: str, match_text1: str, match_text2: str, match_text3: str) -> str:
    """Create a prompt to rewrite (polish) template_txt."""
    prompt = f"""
You are a clinical documentation assistant. Rewrite the following emergency department (ED) patient description to sound natural, fluent, and clinically appropriate, closely matching the narrative style of the examples.

STYLE REFERENCES (for tone/structure only — do NOT copy wording, numbers, or any placeholder content):
1. "{match_text1}"
2. "{match_text2}"
3. "{match_text3}"

TEXT TO REWRITE:
"{template_txt}"

INSTRUCTIONS:
- Enhance realism through bedside context and functional impact (arrival scene, patient demeanor, brief dialogue snippets, and how symptoms affect daily activities), but do not add any new clinical facts—no invented exam findings, vitals, tests, imaging, diagnoses, or treatments.
- Add only plausible, non-speculative descriptive detail and do not invent exam findings or interventions not provided.
- Preserve all key clinical facts exactly as provided (symptoms, timing, vitals, exam findings, diagnostics, treatments, responses, relevant negatives, disposition, etc.).
- You may expand for readability and clinical flow (e.g., add connecting phrases), but do NOT invent any new clinical details, values, or events.
- Use varied, chart-authentic ED documentation phrasing to create a realistic scene (e.g., rotate equivalent expressions for “at triage,” “vital signs were recorded as,” and “per history,” do not rely only on the specific example phrases provided).
- If a field is missing or incomplete (e.g., truncated BP), do not fabricate it; explicitly mark it as not documented (e.g., “BP 102/—, diastolic not documented”). Preserve the original units as given.
- Ignore and remove any placeholder symbols or separators such as ___, ===, **, and similar; omit them entirely from the output.
- If a data field is missing or blank in the original text, do NOT add commentary like “not measured” or “not available.” Only include information explicitly present.
- Do NOT copy any placeholder data from the examples. Use the examples only to imitate writing style.
- Output ONLY the rewritten narrative. Do not include headings like “Polished version,” and do not add explanations.
"""
    return prompt


def call_deepseek_api(api_key: str, prompt: str, max_retries: int = 100) -> Optional[str]:
    """Call DeepSeek API with retry logic."""
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
            continue
        except requests.exceptions.RequestException as e:
            print(f"  API call error: {e}, retrying {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
        except KeyError as e:
            print(f"  Response parsing error: {e}")
            return None
        except Exception as e:
            print(f"  Unknown error: {e}")
            return None

    print("  All retry attempts failed")
    return None


def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from a JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return []
    return []


def save_single_result(result: Dict, filename: str) -> None:
    """Append a single result into a JSON file (stored as a list)."""
    # Load existing results
    existing_results = load_existing_results(filename)

    # Append the new result
    existing_results.append(result)

    # Save back
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(existing_results, file, indent=2, ensure_ascii=False)

    print(f"  Result saved to: {filename}")


def main():
    # Configuration
    # NOTE: Do NOT hardcode API keys in code. Use environment variables instead.
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key")

    JSON_FILENAME = "your_input_dir/result_matched_part1.json"
    OUTPUT_FILENAME = "your_output_dir/polished_results1.json"

    try:
        # 1) Read JSON data
        print("Reading JSON file...")
        all_data = read_json_data(JSON_FILENAME)
        print(f"Total records read: {len(all_data)}")

        # Load processed results (for resume)
        existing_results = load_existing_results(OUTPUT_FILENAME)
        processed_ids = {result["id"] for result in existing_results if "id" in result}

        skipped_records = []
        failed_records = []
        processed_records = len(existing_results)

        # Rebuild triage counters from existing results
        triage_counters = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for result in existing_results:
            triage = result.get("triage", "")
            if triage in triage_counters:
                triage_counters[triage] += 1

        print(f"Found {processed_records} existing results")

        # 2) Process each record
        for i, record in enumerate(all_data, 1):
            record_id = record.get("id", "N/A")

            # Skip already processed records
            if record_id in processed_ids:
                print(f"Skipping already processed record {i}/{len(all_data)} (ID: {record_id})...")
                continue

            print(f"Processing record {i}/{len(all_data)} (ID: {record_id})...")

            try:
                # Validate triage (must be 1-5)
                triage_value = record.get("Triage", "")
                if not is_valid_triage(triage_value):
                    print(f"  Skipped: Invalid triage value '{triage_value}'")
                    skipped_records.append(record_id)
                    continue

                # Check per-triage processing limit
                if not should_process_record(triage_value, triage_counters):
                    print(f"  Skipped: Reached limit for triage level {triage_value}")
                    skipped_records.append(record_id)
                    continue

                # Fetch required fields
                template_txt = record.get("template_txt", "")
                match_text1 = record.get("match_text1", "")
                match_text2 = record.get("match_text2", "")
                match_text3 = record.get("match_text3", "")

                # Ensure required fields exist
                if not all([template_txt, match_text1, match_text2, match_text3]):
                    print("  Skipped: Missing required fields")
                    skipped_records.append(record_id)
                    continue

                # Create prompt
                prompt = create_polish_prompt(template_txt, match_text1, match_text2, match_text3)

                # Call API
                print("  Calling API to polish text...")
                answer = call_deepseek_api(API_KEY, prompt)

                if answer:
                    result_record = {
                        "id": f"{record_id}",
                        "answer": answer.strip('"').strip(),
                        "triage": triage_value,
                    }

                    # Save immediately (single record)
                    save_single_result(result_record, OUTPUT_FILENAME)

                    # Update counters
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

                # Small delay to avoid rate limiting
                if i < len(all_data):
                    time.sleep(1)

            except Exception as e:
                print(f"  Error processing record {i}: {e}")
                failed_records.append(record_id)
                continue

        # 3) Summary
        print("\nProcessing completed!")
        print(f"Total records: {len(all_data)}")
        print(f"Successfully processed: {processed_records}")
        print(f"Skipped records: {len(skipped_records)}")
        print(f"Failed records: {len(failed_records)}")

        print("\nTriage level processing summary:")
        for triage in ["1", "2", "3", "4", "5"]:
            count = triage_counters[triage]
            limits = {
                "1": "unlimited",
                "2": "unlimited",
                "3": "unlimited",
                "4": "unlimited",
                "5": "unlimited",
            }
            print(f"  Triage {triage}: {count} processed (limit: {limits[triage]})")

        if skipped_records:
            print(
                f"\nSkipped record IDs (first 10): {skipped_records[:10]}"
                f"{'...' if len(skipped_records) > 10 else ''}"
            )

        if failed_records:
            print(
                f"Failed record IDs (first 10): {failed_records[:10]}"
                f"{'...' if len(failed_records) > 10 else ''}"
            )

    except FileNotFoundError:
        print(f"Error: File not found {JSON_FILENAME}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {JSON_FILENAME}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
