# kakaotalk-trained-ai-chatbot

카카오톡 대화 로그 기반 로컬 챗봇 프로젝트.

현재 기본 설계:
- 문맥 윈도우 학습 (`context_windows`)
- 답변 구간 중심 손실 (`response_only`)
- 자동 resume / 안전 중단 유지
- 출력은 익명 2인 대화 UX + 문장만 출력

## 0) 설치 (Git Bash)
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

`.env` 필수:
- `CHATBOT_PASSWORD`
- `CHATBOT_MODEL_KEY` (enc 모델 쓸 때)

## 1) 실행 명령 (인자 없이 기본 사용)
### 1-1. 기존 산출물 아카이브/청소
```bash
python -m chatbot.ops archive
```

### 1-2. raw 정리
```bash
python -m chatbot.ops organize
```

### 1-3. 전처리
```bash
python -m chatbot.ops preprocess
```
생성물:
- `data/processed/train.bin`, `val.bin`
- `data/processed/train_loss_mask.bin`, `val_loss_mask.bin`
- `data/processed/tokenizer.json`, `stats.json`, `preview.json`

### 1-4. 학습 (장기 실행 기본)
```bash
python -m chatbot.ops train
```
- `max_steps`는 config에 매우 크게 고정되어 있음 (장기 러닝)
- `Ctrl+C` 중단 가능
- 재실행하면 자동 resume

### 1-5. 빠른 품질 점검(자동 3턴)
```bash
python -m chatbot.ops smoke
```

### 1-6. 단발/대화
```bash
python -m chatbot.ops reply "오늘 뭐함"
python -m chatbot.ops chat
```

### 1-7. 브리지 테스트/실전
```bash
python -m chatbot.ops bridge --dry
python -m chatbot.ops bridge --send
```

## 2) 하이퍼파라미터 조정 원칙
요청대로 **CLI 인자 조정 없이**, 아래 상수 파일만 수정:
- 학습/재개/손실: `configs/train.yaml`
- 전처리/문맥/필터: `configs/paths.yaml > processed.preprocess`
- 추론/채팅/브리지: `configs/gen.yaml`

핵심 상수:
- `context_turns`, `min_target_chars`, `sample_stride`, `drop_low_signal`
- `optimization.learning_rate`, `lr_decay_steps`, `batch_size`, `grad_accum_steps`
- `objective.loss_mode: response_only`

## 3) 현재 학습 구조 요약
1. 전처리: 대화 세션 분리 + 문맥 N턴 + 타겟 1턴 샘플 생성
2. 토크나이즈: `<|bos|><|spk:...|>...<|eos|>` 형식
3. 손실마스크: 문맥 토큰 0, 답변 구간 1 (`*_loss_mask.bin`)
4. 학습: GPT causal LM + 마스크된 cross-entropy
5. 추론: 최근 히스토리로 프롬프트 구성 -> 1턴 생성 -> prefix 제거/정리 후 출력

## 4) 자동 resume / 중단
- 자동 resume: `checkpoints/<run_name>/latest.pt` 있으면 이어서 학습
- 중단:
  - `Ctrl+C`
  - `checkpoints/<run_name>/STOP` 파일 생성
- 저장:
  - `latest.pt`, `best.pt`, `snapshots/step_xxxxxx.pt`

## 5) 보안/업로드
- 추론/브리지는 `.env` 비밀번호 없으면 실행 차단
- 민감 데이터(`data/raw`, `data/processed`, `checkpoints`)는 gitignore
- 공개는 `artifacts/model_latest.pt` 또는 `.enc` 1개만 권장
- 퍼블리시:
```bash
python -m chatbot.ops publish
python -m chatbot.ops publish --encrypt
```
