export CUDA_VISIBLE_DEVICES=0
python your_script_dir/bert/pre.py \
  --test_files \
    your_output_dir/triage_embed_soap_s.jsonl \
    your_output_dir/triage_embed_soap_o.jsonl \
    your_output_dir/triage_embed_soap_a.jsonl \
    your_output_dir/triage_embed_soap_p.jsonl \
  --stage1_ckpt your_ckpt_dir/stage1_binary_ep045_val0.7979_test0.7992.pt \
  --stage2_high_ckpt your_ckpt_dir/stage2_high_ep077_val0.8259_test0.8242.pt \
  --stage2_low_ckpt your_ckpt_dir/stage2_low_ep058_val0.8258_test0.8269.pt \
  --output_path your_output_dir/pred_with_weights.jsonl \
  --save_named_weights
