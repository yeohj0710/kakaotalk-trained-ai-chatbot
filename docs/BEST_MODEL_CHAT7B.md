# Best Model Handoff

This document is the canonical handoff for the current best conversational model in this repository.

Use this first in a new session.

## 1. Current Best Model

- Purpose: one-on-one Korean chat that behaves like a KakaoTalk room member, but replies directly to the user
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Persona target: `박현탁`
- Best run: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- Best adapter dir: `checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best`
- Best training metric: `3.303590774536133`
- Best training checkpoint: `checkpoint-250`
- Best runtime inference config: `configs/sft.chat7b.pht.chat.yaml`

Canonical interactive command:

```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

Canonical smoke command:

```bash
python -m chatbot.sft_ops smoke --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## 2. Why This Model Won

The best result did not come from the largest model or the most aggressive refine.

The practical winning recipe was:
- switch from room-next-turn imitation to `projected_dialogue`
- use an instruct base model
- train a single-persona one-on-one chat bot
- keep inference stricter than training
- stop before over-constraining the outputs

The production stack is split in two parts:
- training identity and data shaping: `configs/sft.chat7b.pht.yaml` and `configs/sft.chat7b.pht.refine.yaml`
- runtime decoding and prompt shaping: `configs/sft.chat7b.pht.chat.yaml`

The training config that produced the adapter is not the same as the best runtime config. That is intentional.

## 3. Training Lineage

This is the lineage that led to the current best model.

### 3.1 Base persona chat run

- Run: `room_chat_qwen25_7b_instruct_pht_v1`
- Config: `configs/sft.chat7b.pht.yaml`
- Init mode: fresh LoRA on `Qwen/Qwen2.5-7B-Instruct`
- Best metric: `3.3140006065368652`
- Best checkpoint: `checkpoint-4750`

### 3.2 Main refine run

- Run: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- Config: `configs/sft.chat7b.pht.refine.yaml`
- Init adapter: `checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1/adapter_best`
- Best metric: `3.303590774536133`
- Best checkpoint: `checkpoint-250`
- Status: current best production model

### 3.3 Experimental runs that should not be used as production defaults

`room_chat_qwen25_7b_instruct_pht_v1_answer_refine`
- Goal: bias toward reply-target samples
- Best metric: `3.3496906757354736`
- Result: more brittle, degraded practical replies

`room_chat_qwen25_7b_instruct_pht_v1_direct_refine`
- Goal: bias toward short direct answers
- Best metric: `3.400972843170166`
- Result: often too narrow and context-thin

`room_chat_qwen25_7b_instruct_pht_v1_closer_refine`
- Goal: make answers close more cleanly
- Best metric: `3.4420717064848633`
- Result: further regression

Rule:
- keep these experimental runs for reference
- do not treat them as the default model unless a future session re-tests and proves otherwise

## 4. Data and Design Decisions That Mattered

### 4.1 What changed from the old pipeline

The older room model learned "the next message in the room".

The winning chat model instead learns:
- user context projected into `user`
- target persona projected into `assistant`
- one-on-one continuation, not generic room continuation

This is implemented through:
- `data.training_mode: projected_dialogue`
- `data.target_speakers: ["박현탁"]`
- `project_user_names: true`
- `use_chat_template: true`

### 4.2 Persona choice

`박현탁` was selected because:
- enough message volume to train on
- lower noise than the highest-volume room speakers
- better fit for direct reply behavior than pure room-hype speakers

This persona is not "the smartest member".
It is the best tradeoff found between:
- enough samples
- enough conversational directness
- manageable noise

### 4.3 What failed

These patterns tended to make the model worse:
- training for "short directness" too aggressively
- narrowing the target distribution until answers became thin or repetitive
- assuming lower loss on a narrower filtered set automatically means better chat quality

Practical lesson:
- after `pht_v1_refine`, further quality did not come from tighter filtering alone
- future gains now require better correction data, not more of the same refine

## 5. Canonical Files

### 5.1 Production inference

- `configs/sft.chat7b.pht.chat.yaml`

Use this for actual chat tests with the best adapter.

Why:
- shorter max output
- lower temperature
- higher repetition control
- stronger prompt against filler, lists, and multiline fragments

### 5.2 Training files that produced the best adapter

- `configs/sft.chat7b.pht.yaml`
- `configs/sft.chat7b.pht.refine.yaml`
- `scripts/train_chat7b_pht.ps1`

### 5.3 Core implementation files

- Preprocess: `src/chatbot/sft_preprocess.py`
- Config loader/defaults: `src/chatbot/sft_config.py`
- SFT training: `src/chatbot/sft_train.py`
- Inference/decoding: `src/chatbot/sft_infer.py`
- Chat CLI: `src/chatbot/sft_chat.py`
- Unified ops: `src/chatbot/sft_ops.py`

## 6. How To Reproduce The Best Model

Use this if the best run needs to be rebuilt from scratch.

Preprocess:

```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.chat7b.pht.yaml --env_path .env
```

Base SFT:

```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1
```

Refine:

```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1/adapter_best
```

Single command script:

```bash
powershell.exe -ExecutionPolicy Bypass -File "./scripts/train_chat7b_pht.ps1"
```

Resume rule:
- rerun the same command
- checkpoints under the same `run_name` are picked up automatically

## 7. How To Use The Best Model In Later Sessions

When a new session starts, the correct mental model is:

1. Production model is `room_chat_qwen25_7b_instruct_pht_v1_refine`
2. Production inference config is `configs/sft.chat7b.pht.chat.yaml`
3. Experimental refines after that are not the default
4. If a task says "use the best model", it means the pair above

If you need to wire this into other features:
- use `--run_name room_chat_qwen25_7b_instruct_pht_v1_refine`
- prefer `configs/sft.chat7b.pht.chat.yaml` for chat, reply, smoke, and serve
- avoid switching to `direct_refine` or `closer_refine` unless explicitly re-evaluating experiments

## 8. Practical Tips

- Training config and inference config are allowed to differ.
- Real chat quality matters more than `eval_loss` across mismatched datasets.
- Compare metrics only within the same task/data regime.
- `projected_dialogue` was the major structural improvement.
- `chat_template` alignment between training and inference is required.
- If future work stalls, collect failed prompts and build a correction dataset instead of blindly extending refine.

## 8.1 Correction Experiments After The Best Model

The following correction runs were built and tested after `room_chat_qwen25_7b_instruct_pht_v1_refine`, but none of them replaced production:

- `room_chat_qwen25_7b_instruct_pht_v1_correction_v1`
  - type: correction-only refine
  - result: rejected after qualitative review due to unstable outputs
- `room_chat_qwen25_7b_instruct_pht_v1_correction_mix_v1`
  - type: reviewed corrections + persona anchors
  - result: safer than correction-only, but still worse than production on real prompt tests
- `room_chat_qwen25_7b_instruct_pht_v1_correction_mix_v2`
  - type: larger reviewed corrections + persona anchors
  - result: still failed promotion

Promotion rule:
- do not promote correction models by correction-set `eval_loss` alone
- require repeated qualitative wins on the same real prompt batch

## 9. Recommended Next-Step Directions

If future work continues from this best model, the highest-value directions are:
- build a small correction dataset from actual failed conversations
- separate room style from direct answer more cleanly in training data
- keep persona fixed unless there is a strong reason to re-run speaker selection

Do not start by:
- reviving `answer_refine`
- reviving `direct_refine`
- reviving `closer_refine`
- switching to 14B without a new data plan
