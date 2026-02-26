# kakaotalk-trained-ai-chatbot (LoRA SFT)

이 프로젝트는 카카오톡 대화 로그를 이용해 **사전학습 LLM(Qwen 계열)**에 LoRA 추가학습을 수행하는 파이프라인입니다.

목표:
- from-scratch가 아니라, 이미 언어능력이 있는 모델에 톡방 말투/반응을 특화
- 실행은 최대한 단순하게
- 중단/재개/최적 체크포인트(best) 운영

## 1) 설치 (Git Bash)
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

`.env` 필수:
- `CHATBOT_PASSWORD=...`

## 2) 기본 실행 순서
```bash
python -m chatbot.sft_ops archive
python -m chatbot.sft_ops organize
python -m chatbot.sft_ops preprocess
python -m chatbot.sft_ops train
```

또는 스크립트:
```bash
./scripts/run.sh preprocess
./scripts/run.sh train
```

PowerShell:
```powershell
.\scripts\run.ps1 preprocess
.\scripts\run.ps1 train
```

## 3) 중단/재개
- 중단: `Ctrl+C`
- 재개: 다시 `python -m chatbot.sft_ops train`
- STOP 파일 방식: `checkpoints_lora/<run_name>/STOP`

자동 재개는 `checkpoints_lora/<run_name>/checkpoint-*`에서 수행됩니다.

## 4) 추론/테스트
단발:
```bash
python -m chatbot.sft_ops reply "오늘 뭐함"
```

대화:
```bash
python -m chatbot.sft_ops chat
```

스모크(고정 3턴):
```bash
python -m chatbot.sft_ops smoke
```

## 5) 체크포인트 구조
- 학습 체크포인트: `checkpoints_lora/<run_name>/checkpoint-*`
- 최신 어댑터: `checkpoints_lora/<run_name>/adapter_latest`
- 최고 어댑터: `checkpoints_lora/<run_name>/adapter_best`
- 상태 파일: `checkpoints_lora/<run_name>/status.json`

추론은 기본적으로 `adapter_best`를 우선 사용합니다.

## 6) 핵심 설정 파일 (상수 기반)
- `configs/sft.yaml` 하나로 관리

중요 항목:
- `model.base_model`: 베이스 LLM
- `data.context_turns`, `data.min_target_chars`, `data.drop_low_signal`
- `training.max_steps`, `training.learning_rate`, `training.eval_steps/save_steps`
- `generation.temperature/top_p/top_k/repetition_penalty`

요청사항대로, 기본 사용 시 CLI 인자로 하이퍼파라미터를 바꾸지 않습니다.

## 7) 품질 관련 현실적 포인트
- 성능은 **데이터 품질 + 베이스모델 품질** 영향이 큽니다.
- 너무 짧은 답이 많으면 `data.min_target_chars`를 올리세요.
- 말이 깨지면 `generation.temperature`를 낮추고 `top_k/top_p`를 보수적으로 조정하세요.

## 8) 보안
- `CHATBOT_PASSWORD` 없으면 추론/채팅 실행 차단
- 민감 데이터는 기본적으로 gitignore 대상:
  - `data/raw/*`
  - `data/sft/*`
  - `checkpoints_lora/*`

## 9) 주의
- 처음 학습 시 베이스 모델 다운로드가 필요합니다(Hugging Face).
- GPU/VRAM 상황에 따라 `model.load_in_4bit`가 실패하면 자동으로 비양자화 로딩으로 폴백될 수 있습니다.
  - 이 경우 VRAM 사용량이 크게 증가합니다.
