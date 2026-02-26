# kakaotalk-trained-ai-chatbot (CPT + SFT LoRA)

카카오톡 대화 로그를 기반으로, **베이스 LLM(Qwen2.5-3B)**에 2단계로 학습합니다.

- 1단계: **CPT** (raw 대화 흐름 next-token 학습)
- 2단계: **SFT** (문맥 -> 다음 답변 학습)

목표는 톡방 말투/리듬을 최대한 반영한 대화형 모델입니다.

## 1) 설치 (Git Bash)
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

`.env` 필수:
- `CHATBOT_PASSWORD=...`

## 2) 기본 실행
```bash
python -m chatbot.sft_ops organize
python -m chatbot.sft_ops preprocess
python -m chatbot.sft_ops train
```

`train`은 자동으로:
1. CPT 단계 실행/재개
2. CPT 완료 시 SFT 단계 실행/재개

## 3) 학습 중단/재개
- 중단: `Ctrl+C`
- 재개: 동일 명령 재실행
```bash
python -m chatbot.sft_ops train
```
- STOP 파일: `checkpoints_lora/<run_name>/STOP`

## 4) 체크포인트 정책 (기본)
- 저장(save): **자주** (`save_steps: 500`)
- 검증(eval): **드물게**  
  - CPT: `eval_steps: 8000`
  - SFT: `eval_steps: 5000`

즉, 테스트용 모델은 자주 갱신되고 검증 오버헤드는 줄어듭니다.

## 5) 테스트
단발:
```bash
python -m chatbot.sft_ops reply "오늘 뭐함"
```

대화:
```bash
python -m chatbot.sft_ops chat
```

최신 체크포인트로 명시 테스트:
```bash
LATEST="$(ls -d checkpoints_lora/room_lora_qwen25_3b_base/checkpoint-* 2>/dev/null | sort -V | tail -1)"
python -m chatbot.sft_ops chat --adapter "$LATEST"
```

로컬 HTTP API 서버(웹 연동용):
```bash
python -m chatbot.sft_ops serve --host 127.0.0.1 --port 8000 --config_sft configs/sft.yaml --env_path .env
```

## 6) 핵심 설정 파일
- `configs/sft.yaml` (상수 기반 단일 설정)

중요 항목:
- `model.base_model`
- `pipeline.*` (CPT -> SFT 흐름)
- `cpt_training.*`, `training.*` (저장/검증 주기 포함)
- `generation.*`

## 7) 보안/민감데이터
- 비밀번호 게이트: `CHATBOT_PASSWORD`
- git 커밋 금지 대상:
  - `data/raw/*`
  - `data/sft/*`
  - `checkpoints_lora/*`

## 8) 참고
- `load_in_4bit=true`인데 `bitsandbytes`가 없으면 full-precision 폴백될 수 있습니다.
- 이 경우 속도/VRAM 사용량이 크게 늘 수 있습니다.
