export CUDA_VISIBLE_DEVICES=0
python your_script_dir/train.py \
  --train_files \
    your_train_dir/triage_embed_soap_s.jsonl \
    your_train_dir/triage_embed_soap_o.jsonl \
    your_train_dir/triage_embed_soap_a.jsonl \
    your_train_dir/triage_embed_soap_p.jsonl \
  --model_type_stage1 bert --model_type_stage2_high bert --model_type_stage2_low bert \
  --norm none \
  --s1_gate_temperature 1.25 --s1_gate_entropy_lambda 0.01 --s1_gate_balance_lambda 0.9 --s1_gate_diversity_lambda 0.5 \
  --h_gate_temperature 1.0 --h_gate_entropy_lambda 0.06 --h_gate_balance_lambda 1.0 --h_gate_diversity_lambda 0.5 \
  --l_gate_temperature 1.0 --l_gate_entropy_lambda 0.03 --l_gate_balance_lambda 1.0 --l_gate_diversity_lambda 0.5 \
  --lr 3e-5 --weight_decay 5e-4 \
  --bert_layers 2 --bert_heads 8 --bert_ffn_dim 1024 \
  --epochs 200 --batch_size 128 \
  --save_dir your_ckpt_dir
