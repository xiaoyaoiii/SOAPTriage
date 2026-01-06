#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Run SOAP four encoding modes sequentially: s/o/a/p"

# -------------------------
# Fixed parameters
# -------------------------
script_path="your_script_dir/soap_encoding.py"
input_json="your_data_dir/emnlp.json"
model_path="your_model_dir/triage_model_8b"

max_length=2048
batch_size=16
eval_layer=-1
force_device="cuda:2"

# -------------------------
# Output files (the paths you specified)
# -------------------------
declare -A output_jsonl_map
output_jsonl_map["s"]="your_output_dir/triage_embed_soap_s.jsonl"
output_jsonl_map["o"]="your_output_dir/triage_embed_soap_o.jsonl"
output_jsonl_map["a"]="your_output_dir/triage_embed_soap_a.jsonl"
output_jsonl_map["p"]="your_output_dir/triage_embed_soap_p.jsonl"

soap_modes=("s" "o" "a" "p")

# -------------------------
# Basic checks
# -------------------------
if [ ! -f "$script_path" ]; then
  echo "[ERROR] Script not found: $script_path"
  exit 1
fi
if [ ! -f "$input_json" ]; then
  echo "[ERROR] Input data not found: $input_json"
  exit 1
fi
if [ ! -d "$model_path" ]; then
  echo "[ERROR] Model directory not found: $model_path"
  exit 1
fi

# Do NOT force CUDA_VISIBLE_DEVICES to avoid mapping conflicts with --force_device cuda:2
echo "[INFO] force_device = ${force_device}"
echo "[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES-<not set>}"

total=${#soap_modes[@]}
idx=1

for soap_mode in "${soap_modes[@]}"; do
  output_jsonl="${output_jsonl_map[$soap_mode]}"

  if [ -z "${output_jsonl:-}" ]; then
    echo "[ERROR] output_jsonl is not configured for soap_mode=${soap_mode}"
    exit 1
  fi

  output_dir="$(dirname "$output_jsonl")"
  mkdir -p "$output_dir"
  log_file="${output_dir}/log_${soap_mode}.txt"

  echo ""
  echo "================================================================="
  echo "[TASK $idx/$total] SOAP mode: $soap_mode"
  echo "Output: $output_jsonl"
  echo "Log: $log_file"
  echo "================================================================="

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
    echo "================================================================="
    echo "[START] $(date)"
    echo "soap_mode: $soap_mode"
    echo "output_jsonl: $output_jsonl"
    echo "command:"
    printf '%q ' "${cmd[@]}"
    echo ""
    echo "================================================================="
  } > "$log_file"

  # Run and record logs, while preserving the real exit code
  "${cmd[@]}" 2>&1 | tee -a "$log_file"
  exit_status=${PIPESTATUS[0]}

  {
    echo ""
    echo "[END] $(date)"
    echo "exit_status: $exit_status"
    if [ "$exit_status" -eq 0 ]; then
      echo "[SUCCESS] Done: soap_mode=$soap_mode"
    else
      echo "[ERROR] Failed: soap_mode=$soap_mode"
    fi
  } >> "$log_file"

  if [ "$exit_status" -ne 0 ]; then
    echo "[FATAL] Failed in the middle, stopping remaining tasks. Failed mode: $soap_mode"
    exit "$exit_status"
  fi

  idx=$((idx + 1))
done

echo ""
echo "================================================================="
echo "[INFO] All four SOAP encoding tasks have finished sequentially"
echo "================================================================="
