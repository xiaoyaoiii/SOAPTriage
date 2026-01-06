import os
import json
import torch
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

PoolMethod = Literal["mean", "mean_including_padding", "last_hidden_state", "first_hidden_state"]
LayerMixing = Literal["none", "avg", "exp"]


@dataclass
class ChatEmbeddingConfig:
    model_path: str
    max_length: int = 2048
    evaluation_layer_idx: int = -1
    layer_window: int = 0
    layer_mixing: LayerMixing = "avg"          # none | avg | exp
    layer_mixing_tau: float = 1.0              # temperature for exp mixing (smaller -> more weight on later layers)
    include_embedding_in_mixing: bool = False  # allow embedding layer (index=0) in mixing
    pooling_method: PoolMethod = "mean"        # mean | mean_including_padding | last_hidden_state | first_hidden_state
    batch_size_hint: int = 1
    return_full_prompt_embed: bool = True
    return_all_layer_pooled: bool = False
    normalize_embedding: bool = True
    fp16: bool = True
    device_map: str = "auto"
    force_device: Optional[str] = None         # e.g. "cuda:0" to force single device
    add_generation_prompt: bool = False        # append generation prompt to full prompt (default off)
    soap_mode: str = "none"                    # none | s | o | a | p


def safe_print(*a, **kw):
    try:
        print(*a, **kw)
    except UnicodeEncodeError:
        print(*(str(x).encode("utf-8", "ignore").decode("utf-8") for x in a), **kw)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))


def find_optimal_batch_size(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_length: int,
    start_batch: int = 1,
    max_trials: int = 8,
    add_generation_prompt: bool = False,
) -> int:
    """
    Probe batch size with 1,2,4,... until OOM and rollback.
    To reduce false positives, only probe on up to the first 32 samples.
    """
    test_samples = messages_list[: min(len(messages_list), 32)]
    batch_size = start_batch
    had_success = False

    for _ in range(max_trials):
        try:
            batch_subset = test_samples[:batch_size]
            prompts_full = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
                for msgs in batch_subset
            ]
            enc = tokenizer(
                prompts_full,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            device = model.get_input_embeddings().weight.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )

            had_success = True
            if batch_size >= len(test_samples):
                break
            batch_size *= 2

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                safe_print(f"[BatchSizeProbe] OOM at batch={batch_size}, rolling back.")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                break
            raise e

    return max(1, batch_size)


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    method: PoolMethod = "mean",
) -> torch.Tensor:
    """
    Supports left padding.
    hidden_states: [B, L, H]
    attention_mask: [B, L]
    """
    mask = attention_mask.to(hidden_states.device).unsqueeze(-1).float()
    B, L, H = hidden_states.shape

    if method == "mean":
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (hidden_states * mask).sum(dim=1) / denom

    if method == "mean_including_padding":
        return hidden_states.mean(dim=1)

    if method == "last_hidden_state":
        valid_len = attention_mask.sum(dim=1)          # [B]
        Lm = attention_mask.size(1)
        pad_full = (Lm - valid_len)                    # [B]
        idx_pos = pad_full + valid_len - 1             # [B]
        idx = idx_pos.view(-1, 1, 1).expand(-1, 1, H)  # [B,1,H]
        return torch.gather(hidden_states, dim=1, index=idx).squeeze(1)

    if method == "first_hidden_state":
        valid_len = attention_mask.sum(dim=1)          # [B]
        Lm = attention_mask.size(1)
        pad_full = (Lm - valid_len)
        idx_pos = pad_full.clamp(min=0)
        idx = idx_pos.view(-1, 1, 1).expand(-1, 1, H)
        return torch.gather(hidden_states, dim=1, index=idx).squeeze(1)

    raise ValueError(f"Unsupported pooling method: {method}")


def _mix_layers(
    hidden_states_list: List[torch.Tensor],
    eval_idx: int,
    window: int,
    mode: LayerMixing,
    tau: float = 1.0,
    include_embedding: bool = False,
) -> torch.Tensor:
    """
    Mix multiple [B,L,H] hidden state layers into a single [B,L,H].

    - eval_idx: selected layer index (0=embeddings, 1..num_layers=block outputs)
    - window: lookback window size (inclusive of eval_idx)
    - mode:
      - none: use eval layer only
      - avg: uniform average
      - exp: exponential weights favoring later layers
    - include_embedding: allow layer 0 (embeddings) in mixing
    """
    low = 0 if include_embedding else 1
    start = max(low, eval_idx - window)
    end = eval_idx

    selected = hidden_states_list[start : end + 1]
    if mode == "none" or len(selected) == 1:
        return hidden_states_list[eval_idx]

    stack = torch.stack(selected, dim=0)  # [K,B,L,H]

    if mode == "avg":
        return stack.mean(dim=0)

    if mode == "exp":
        K = stack.shape[0]
        distances = torch.arange(K, device=stack.device).float()
        weights = torch.exp(-distances / max(1e-6, tau))  # [K]
        weights = (weights / weights.sum()).view(K, 1, 1, 1)
        return (stack * weights).sum(dim=0)

    raise ValueError(f"Unsupported layer mixing mode: {mode}")


class ChatEmbedder:
    def __init__(self, cfg: ChatEmbeddingConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.config = AutoConfig.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        self.total_layers = getattr(self.config, "num_hidden_layers", None)
        if self.total_layers is None:
            self.total_layers = len(getattr(self.config, "hidden_layers", [])) - 1
        if self.total_layers is None or self.total_layers <= 0:
            raise ValueError("Unable to determine total_layers from model config.")

        if cfg.evaluation_layer_idx == -1:
            self.evaluation_layer_idx = self.total_layers
        else:
            self.evaluation_layer_idx = cfg.evaluation_layer_idx

        if not (0 <= self.evaluation_layer_idx <= self.total_layers):
            raise ValueError(
                f"evaluation_layer_idx={self.evaluation_layer_idx} out of range; total_layers={self.total_layers}"
            )

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
        )
        if cfg.force_device:
            model_kwargs["device_map"] = None
        else:
            model_kwargs["device_map"] = cfg.device_map

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)
        if cfg.force_device:
            self.model.to(cfg.force_device)

        self.model.eval()

    def build_messages(self, text: str) -> List[Dict[str, str]]:
        """
        Build chat messages based on cfg.soap_mode:
        - none: triage-focused system prompt
        - s/o/a/p: SOAP-focused prompts
        """
        mode = (self.cfg.soap_mode or "none").lower()

        if mode == "none":
            return [
                {
                    "role": "system",
                    "content": "You are an emergency triage assistant. Determine the ESI triage level based on the medical record summary.",
                },
                {"role": "user", "content": text},
            ]

        system_msg = "You are an emergency clinician using the SOAP framework."

        if mode == "s":
            user_msg = f"""Below is a short triage note describing a patient's condition:

{text}

Task:
1. Extract and rewrite only the SUBJECTIVE part (S in SOAP): what the patient or caregiver reports, including main complaint, onset, context, and perceived severity.
2. Ignore physical exam findings, vital signs, objective measurements, and any management plan.
3. Write 1–2 concise sentences in English.

Output ONLY the rewritten subjective description, with no labels or extra commentary.
"""
            return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

        if mode == "o":
            user_msg = f"""Below is a short triage note describing a patient's condition:

{text}

Task:
1. Extract and rewrite only the OBJECTIVE part (O in SOAP): vital signs, observable findings, measurements, and documented exam facts.
2. Do NOT include patient-reported symptoms unless they are clearly measured/observed.
3. Write 1–2 concise sentences in English.

Output ONLY the rewritten objective description, with no labels or extra commentary.
"""
            return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

        if mode == "a":
            system_msg_a = (
                "You are an emergency clinician using the SOAP framework for ESI triage (Emergency Severity Index, levels 1–5)."
            )
            user_msg = f"""Below is a short triage note describing a patient's condition:

{text}

Task:
1. Infer and summarize the ASSESSMENT (A in SOAP): likely problem(s), differential considerations, and severity/risk.
2. Use 1–3 concise sentences in English emphasizing risk (stable vs potentially life-threatening).
3. Do NOT mention any specific ESI level number.

Output ONLY the assessment, with no labels or extra commentary.
"""
            return [{"role": "system", "content": system_msg_a}, {"role": "user", "content": user_msg}]

        if mode == "p":
            system_msg_p = "You are an emergency clinician using the SOAP framework."
            user_msg = f"""Below is a short triage note describing a patient's condition:

{text}

Task:
1. Infer and summarize the PLAN (P in SOAP): likely ED actions/resources needed (e.g., labs, imaging, meds, monitoring, procedures, observation).
2. Use 1–3 concise sentences in English.
3. Do NOT mention any specific ESI level number.

Output ONLY the plan, with no labels or extra commentary.
"""
            return [{"role": "system", "content": system_msg_p}, {"role": "user", "content": user_msg}]

        return [
            {
                "role": "system",
                "content": "You are an emergency triage assistant. Determine the ESI triage level based on the medical record summary.",
            },
            {"role": "user", "content": text},
        ]

    def _tokenize_triplet(self, messages_batch: List[List[Dict[str, str]]]):
        tokenizer = self.tokenizer
        max_length = self.cfg.max_length

        prompts_full = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=self.cfg.add_generation_prompt)
            for msgs in messages_batch
        ]
        prompts_su = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_batch
        ]
        user_texts = [msgs[-1]["content"] for msgs in messages_batch]

        enc_full = tokenizer(prompts_full, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc_su = tokenizer(prompts_su, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc_user = tokenizer(user_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return enc_full, enc_su, enc_user

    def _move_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.model.get_input_embeddings().weight.device
        return input_ids.to(device), attention_mask.to(device)

    def encode_batch(self, messages_batch: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        enc_full, enc_su, enc_user = self._tokenize_triplet(messages_batch)
        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        input_ids, attention_mask = self._move_inputs(input_ids, attention_mask)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states_list = list(outputs.hidden_states)

            mixed_hidden = _mix_layers(
                hidden_states_list=hidden_states_list,
                eval_idx=self.evaluation_layer_idx,
                window=max(0, int(self.cfg.layer_window)),
                mode=self.cfg.layer_mixing,
                tau=float(self.cfg.layer_mixing_tau),
                include_embedding=bool(self.cfg.include_embedding_in_mixing),
            )

            if self.cfg.return_full_prompt_embed:
                prompt_pooled = pool_hidden_states(mixed_hidden, attention_mask, self.cfg.pooling_method)
                if self.cfg.normalize_embedding:
                    prompt_pooled = _l2_normalize(prompt_pooled)
                prompt_embeds = prompt_pooled.float().cpu().tolist()
            else:
                prompt_embeds = None

            B, L = input_ids.size()
            su_len = enc_su["attention_mask"].sum(dim=1).tolist()
            user_len = enc_user["attention_mask"].sum(dim=1).tolist()
            full_len = attention_mask.sum(dim=1).tolist()

            user_embeds = []
            for i in range(B):
                seq_full = int(full_len[i])
                seq_su = int(su_len[i])
                k_user = int(user_len[i])

                pad_full = L - seq_full
                su_start_in_full = pad_full + (seq_full - seq_su)
                user_start = su_start_in_full + max(0, seq_su - k_user)
                user_end = su_start_in_full + seq_su

                user_start = max(pad_full, min(user_start, pad_full + seq_full - 1))
                user_end = max(user_start + 1, min(user_end, pad_full + seq_full))

                user_slice = mixed_hidden[i, user_start:user_end, :]
                if user_slice.shape[0] == 0:
                    user_slice = mixed_hidden[i, pad_full : pad_full + seq_full, :][-1:].contiguous()

                pooled_user = user_slice.mean(dim=0)
                if self.cfg.normalize_embedding:
                    pooled_user = _l2_normalize(pooled_user.unsqueeze(0)).squeeze(0)
                user_embeds.append(pooled_user.float().cpu().tolist())

            layer_window_embeds = None
            if self.cfg.layer_window > 0 or self.cfg.layer_mixing != "none":
                window_prompt_pooled = pool_hidden_states(mixed_hidden, attention_mask, self.cfg.pooling_method)
                if self.cfg.normalize_embedding:
                    window_prompt_pooled = _l2_normalize(window_prompt_pooled)
                layer_window_embeds = window_prompt_pooled.float().cpu().tolist()

            all_layer_pooled = None
            if self.cfg.return_all_layer_pooled:
                all_layer_pooled = []
                for hs in hidden_states_list:
                    pooled_layer = pool_hidden_states(hs, attention_mask, self.cfg.pooling_method)
                    if self.cfg.normalize_embedding:
                        pooled_layer = _l2_normalize(pooled_layer)
                    all_layer_pooled.append(pooled_layer.float().cpu().tolist())

        return {
            "user_embeds": user_embeds,
            "prompt_embeds": prompt_embeds,
            "layer_window_embeds": layer_window_embeds,
            "all_layer_pooled": all_layer_pooled,
        }


def load_json_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def process_dataset(
    cfg: ChatEmbeddingConfig,
    input_json: str,
    output_jsonl: str,
    batch_size: Optional[int] = None,
) -> None:
    data_list = load_json_items(input_json)
    embedder = ChatEmbedder(cfg)

    samples = []
    for item in data_list:
        item_id = item.get("id")
        answer = item.get("answer", "")
        label = item.get("triage")
        if not (item_id and answer and label):
            continue
        try:
            label_int = int(label)
        except Exception:
            continue
        messages = embedder.build_messages(answer)
        samples.append((item_id, label_int, messages))

    if len(samples) == 0:
        safe_print("No valid samples found. Exiting.")
        return

    if batch_size is None:
        batch_size = find_optimal_batch_size(
            embedder.model,
            embedder.tokenizer,
            [s[2] for s in samples],
            cfg.max_length,
            start_batch=cfg.batch_size_hint,
            add_generation_prompt=cfg.add_generation_prompt,
        )
    safe_print(f"Using batch_size={batch_size}")

    output_records = []
    saved = 0
    total = 0

    for i in range(0, len(samples), batch_size):
        batch_slice = samples[i : i + batch_size]
        messages_batch = [x[2] for x in batch_slice]
        try:
            batch_result = embedder.encode_batch(messages_batch)
        except RuntimeError as e:
            safe_print(f"[batch {i}] forward pass failed: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            safe_print(f"[batch {i}] unexpected error: {e}")
            continue

        for j, (item_id, label_int, _) in enumerate(batch_slice):
            user_vec = batch_result["user_embeds"][j]
            record = {
                "id": item_id,
                "label": label_int,
                "feature": {"user_embed": user_vec},
            }
            if batch_result["prompt_embeds"] is not None:
                record["feature"]["prompt_embed"] = batch_result["prompt_embeds"][j]
            if batch_result["layer_window_embeds"] is not None:
                record["feature"]["layer_window_embed"] = batch_result["layer_window_embeds"][j]
            output_records.append(record)
            saved += 1

        total += len(batch_slice)
        if saved % 50 == 0:
            safe_print(f"Processed {total} samples, saved {saved} records.")

    save_jsonl(output_jsonl, output_records)
    safe_print(f"Done: read {total} samples, saved {saved} records -> {output_jsonl}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help='e.g. "/path/to/your/INPUT.json"')
    parser.add_argument("--output_jsonl", type=str, required=True, help='e.g. "/path/to/your/OUTPUT.jsonl"')
    parser.add_argument("--model_path", type=str, required=True, help='e.g. "/path/to/your/MODEL_DIR_OR_HF_ID"')
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_layer", type=int, default=-1)
    parser.add_argument("--layer_window", type=int, default=0)
    parser.add_argument("--layer_mixing", type=str, default="avg", choices=["none", "avg", "exp"])
    parser.add_argument("--layer_mixing_tau", type=float, default=1.0)
    parser.add_argument("--include_embedding_in_mixing", action="store_true")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "mean_including_padding", "last_hidden_state", "first_hidden_state"],
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_prompt_embed", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Apply L2 normalization to output embeddings.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 to reduce memory and speed up.")
    parser.add_argument(
        "--force_device",
        type=str,
        default=None,
        help='e.g. "cuda:0" to force model + inputs onto a single device (overrides device_map).',
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Append generation prompt to the full prompt (default off).",
    )
    parser.add_argument(
        "--soap_mode",
        type=str,
        default="none",
        choices=["none", "s", "o", "a", "p"],
        help="SOAP view: none uses base triage prompt; s/o/a/p use SOAP-specific prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ChatEmbeddingConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        evaluation_layer_idx=args.eval_layer,
        layer_window=args.layer_window,
        layer_mixing=args.layer_mixing,
        layer_mixing_tau=args.layer_mixing_tau,
        include_embedding_in_mixing=args.include_embedding_in_mixing,
        pooling_method=args.pooling,
        return_full_prompt_embed=not args.no_prompt_embed,
        normalize_embedding=args.normalize,
        fp16=args.fp16,
        force_device=args.force_device,
        add_generation_prompt=args.add_generation_prompt,
        soap_mode=args.soap_mode,
    )
    process_dataset(cfg, input_json=args.input_json, output_jsonl=args.output_jsonl, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
