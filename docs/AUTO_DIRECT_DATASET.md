# Auto Direct-Answer Dataset

This path is for cases where manual correction review is not practical.

Instead of writing `corrected_reply` row by row, it mines a smaller seed dataset from the existing persona-chat train/val JSONL:

- last visible user turn must look like a question or decision prompt
- target reply must be short, one-line, and non-profane
- target reply must not start with filler like `아니 근데`, `근데`, `그니까`
- target reply must not end with a question

This does **not** replace reviewed correction data.

It is a cheaper intermediate step:
1. keep the current best model as production
2. mine cleaner direct-answer examples from the existing dataset
3. run one small low-LR refine from `room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best`

## Commands

Build the seed dataset:

```bash
python -m chatbot.sft_ops auto_direct_dataset --input_train data/sft/chat7b_pht/train.jsonl --input_val data/sft/chat7b_pht/val.jsonl --train_output data/sft/chat7b_pht_auto_direct/train.jsonl --val_output data/sft/chat7b_pht_auto_direct/val.jsonl --preview_output data/sft/chat7b_pht_auto_direct/preview.json
```

Train the small refine:

```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.auto_direct.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_auto_direct --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best
```

One-shot PowerShell script:

```bash
powershell.exe -ExecutionPolicy Bypass -File "./scripts/train_chat7b_pht_auto_direct.ps1"
```

## Promotion Rule

Do not replace production based only on `eval_loss`.

Promote only if the new run is clearly better on repeated real-prompt tests than:

- run: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- runtime config: `configs/sft.chat7b.pht.chat.yaml`
