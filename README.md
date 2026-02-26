# kakaotalk-trained-ai-chatbot

KakaoTalk TXT 로그로 학습하는 로컬 챗봇 프로젝트입니다.

핵심 목표:
- 기존 명령 체계 유지: `organize / preprocess / train / reply / chat / bridge`
- 기본 UX: 익명 2인 대화(문장만 출력)
- 학습 기본값: 문맥 윈도우 기반(`context_windows`) 전처리

## 1) 설치 (Git Bash 기준)
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

`.env` 필수 항목:
- `CHATBOT_PASSWORD`: 추론/브리지 실행 게이트
- `CHATBOT_MODEL_KEY`: `.enc` 모델 사용 시 복호화 키

## 2) 추천 실행 순서
### A. 기존 산출물 아카이브(초기 리셋)
```bash
python -m chatbot.ops archive --dry
python -m chatbot.ops archive
```
- `data/processed`, `checkpoints`, `artifacts/model_latest*`를 `data/archive/<timestamp>_train_only/`로 이동합니다.
- 원본 raw(`data/raw`)는 기본 유지합니다.

### B. raw 정리
```bash
python -m chatbot.ops organize
```

### C. 전처리 (기본: 문맥 윈도우)
```bash
python -m chatbot.ops preprocess
```
- 기본 모드: `context_windows`
- 설정 위치: `configs/paths.yaml > processed.preprocess`

### D. 학습 (자동 resume)
```bash
python -m chatbot.ops train
```
무한에 가깝게 오래 돌릴 때:
```bash
python -m chatbot.ops train --max_steps 2147483647
```

중단/재개:
- `Ctrl+C`로 중단 가능 (latest 저장)
- 다음 `train` 실행 시 자동 resume
- 또는 `checkpoints/<run_name>/STOP` 파일 생성 시 안전 종료

## 3) 추론/대화
```bash
python -m chatbot.ops reply "오늘 뭐함"
python -m chatbot.ops chat
python -m chatbot.ops bridge --dry
```
- 출력/전송은 기본적으로 role/speaker prefix 없이 문장만 처리됩니다.
- 브리지는 기본적으로 로컬 dry-run 테스트 가능(UI 자동복사 없음).

## 4) Windows PowerShell 래퍼 사용
동일 명령을 유지하면서 PowerShell 래퍼 사용 가능:
```powershell
.\scripts\run.ps1 organize
.\scripts\run.ps1 preprocess
.\scripts\run.ps1 train
.\scripts\run.ps1 reply "테스트"
.\scripts\run.ps1 chat
.\scripts\run.ps1 bridge --dry
```
Git Bash에서 호출 시:
```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 train
```

## 5) 문맥 학습 튜닝 포인트
`configs/paths.yaml`의 `processed.preprocess`:
- `context_turns`: 학습 시 참조할 이전 턴 수
- `min_context_turns`: 최소 문맥 턴
- `session_gap_minutes`: 세션 분리 기준(대화 공백)
- `merge_same_speaker`: 연속 짧은 메시지 병합
- `min_target_chars`: 너무 짧은 타겟 응답 제거
- `drop_low_signal`: ㅋㅋ/ㅠㅠ/기호만 메시지 필터

`configs/train.yaml`:
- `model.block_size`: 컨텍스트 길이 상한
- `optimization.batch_size`, `grad_accum_steps`: VRAM/속도 트레이드오프

## 6) 모델 공개용 단일 아티팩트
```bash
python -m chatbot.ops publish
# 암호화 배포
python -m chatbot.ops publish --encrypt
```
결과:
- `artifacts/model_latest.pt` 또는 `artifacts/model_latest.enc`
- `artifacts/model_info.json`

## 7) 안전 주의
- 민감 데이터(`data/raw`, `data/processed`, `checkpoints`)는 git에 커밋 금지
- 브리지 실전 전송은 반드시 `--send`를 명시할 때만 수행
- 비밀번호/키 게이트는 최소 보호장치이며, 코드 수정 시 우회 가능
