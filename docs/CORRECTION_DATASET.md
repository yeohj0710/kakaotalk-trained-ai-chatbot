# Correction Dataset Workflow

This workflow is for improving the current best model by collecting real failures and turning them into correction data.

Current best model:
- run: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- runtime config: `configs/sft.chat7b.pht.chat.yaml`

## 1. Why this exists

Further blind refine attempts regressed quality.

The remaining path with the best chance of improving quality is:
1. run the best model on real prompts
2. save failures
3. write corrected replies
4. build a small correction dataset
5. run a targeted correction refine

## 2. Input prompt file formats

### Plain text

One prompt per line:

```text
불참실분?
왜요
뭐 먹을지 정해야 돼
```

### JSONL

Use this when you want per-row history or tags:

```json
{"prompt":"불참실분?","tags":["attendance"]}
{"prompt":"왜요","history":[["user","불참실분?"],["bot","나는 안함 ㅋㅋ"]],"tags":["followup"]}
```

Supported fields:
- `prompt` or `user`
- `history`
- `tags`
- `notes`
- `expected`

## 3. Run the best model and collect outputs

```bash
python -m chatbot.sft_ops collect --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one --input artifacts/failure_prompts.txt --output artifacts/failure_runs/pht_v1_refine_run01.jsonl
```

If you want prompts to build on previous generated turns:

```bash
python -m chatbot.sft_ops collect --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one --input artifacts/failure_prompts.jsonl --output artifacts/failure_runs/pht_v1_refine_chain01.jsonl --chain_history
```

## 4. Output format

Each output row includes:
- `prompt`
- `history`
- `reply`
- `tags`
- `notes`
- `expected`
- `corrected_reply`

`corrected_reply` is intentionally left blank so it can be filled in during review.

## 5. What to collect

Good failure buckets:
- question dodging
- weak direct answer
- irrelevant jump
- awkward topic continuation
- repetitive filler
- answer that should close but keeps opening a second thread

## 6. Practical rule

Do not collect only random prompts.

Prefer:
- real chat snippets
- prompts that already exposed weakness
- short multi-turn chains that the model mishandles

## 7. Next step after collection

Once enough rows are reviewed and `corrected_reply` is filled:
- convert reviewed rows into a correction SFT dataset
- run one small correction refine from `room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best`

That next step should be driven by the failure data, not by another blind refine.

## 8. Promotion Criteria

After training a correction model:
- compare it against the production model on the same prompt batch
- reject it if it adds toxicity, topic drift, or unstable new facts
- keep production on `room_chat_qwen25_7b_instruct_pht_v1_refine` unless the new run clearly wins

Practical lesson from current experiments:
- correction-only refine overfit
- anchor-mixed correction refine was safer but still not strong enough to replace production
