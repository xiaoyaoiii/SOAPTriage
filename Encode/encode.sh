#!/usr/bin/env bash
set -euo pipefail

script_path="your_script_dir/soap_encoding.py"
input_json="your_data_dir/MIMIC-IV.json"
model_path="your_model_dir/triage_model_8b"

max_length=2048
batch_size=16
eval_layer=-1
force_device="cuda:2"

declare -A output_jsonl_map=(
  ["s"]="your_output_dir/triage_embed_soap_s.jsonl"
  ["o"]="your_output_dir/triage_embed_soap_o.jsonl"
  ["a"]="your_output_dir/triage_embed_soap_a.jsonl"
  ["p"]="your_output_dir/triage_embed_soap_p.jsonl"
)

soap_modes=("s" "o" "a" "p")

[[ -f "$script_path" ]] || { echo "[ERROR] Script not found: $script_path"; exit 1; }
[[ -f "$input_json"  ]] || { echo "[ERROR] Input not found: $input_json"; exit 1; }
[[ -d "$model_path"  ]] || { echo "[ERROR] Model dir not found: $model_path"; exit 1; }

echo "[INFO] force_device=${force_device} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<not set>}"

total=${#soap_modes[@]}
idx=1

for soap_mode in "${soap_modes[@]}"; do
  output_jsonl="${output_jsonl_map[$soap_mode]:-}"
  [[ -n "$output_jsonl" ]] || { echo "[ERROR] Missing output_jsonl for soap_mode=$soap_mode"; exit 1; }

  output_dir="$(dirname "$output_jsonl")"
  mkdir -p "$output_dir"
  log_file="${output_dir}/log_${soap_mode}.txt"

  echo "=== [$idx/$total] soap_mode=$soap_mode -> $output_jsonl"

  cmd=(
    python "$script_path"
    --input_json "$input_json"
    --output_jsonl "$output_jsonl"
    --model_path "$model_path"
    --max_length "$max_length"
    --batch_size "$batch_size"
    --eval_layer "$eval_layer"
    --normalize
    --fp16
    --force_device "$force_device"
    --soap_mode "$soap_mode"
  )

  {
    echo "[START] $(date)"
    printf 'cmd: '; printf '%q ' "${cmd[@]}"; echo
    echo "----------------------------------------"
  } > "$log_file"

  "${cmd[@]}" 2>&1 | tee -a "$log_file"
  exit_status=${PIPESTATUS[0]}

  {
    echo "----------------------------------------"
    echo "[END] $(date) | exit_status=$exit_status"
  } >> "$log_file"

  [[ "$exit_status" -eq 0 ]] || { echo "[FATAL] Failed: soap_mode=$soap_mode"; exit "$exit_status"; }
  idx=$((idx + 1))
done

echo "[INFO] All SOAP encoding tasks finished."
