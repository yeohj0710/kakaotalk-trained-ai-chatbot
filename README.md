# kakaotalk-trained-ai-chatbot
카카오톡 내보내기 txt로 말투 기반 챗봇을 학습/추론/브리지까지 운영 가능한 형태로 실행하는 프로젝트입니다.

## Quickstart (5분)
```powershell
cd c:\dev\kakaotalk-trained-ai-chatbot
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

`.env`에서 최소 1개는 반드시 설정:
- `CHATBOT_PASSWORD=...` (추론/브리지 게이트)
- `CHATBOT_MODEL_KEY=...` (암호화 모델 사용 시)

CUDA 확인:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 한 줄 실행(통합)
주 엔트리포인트:
- `.\scripts\run.ps1 <command> [args]`
- 내부적으로 `python -m chatbot.ops ...` 호출

명령:
```powershell
.\scripts\run.ps1 organize
.\scripts\run.ps1 preprocess
.\scripts\run.ps1 train
.\scripts\run.ps1 reply "오늘 뭐함"
.\scripts\run.ps1 chat
.\scripts\run.ps1 bridge --dry
.\scripts\run.ps1 bridge --send
```

## 파이프라인
### 1) raw txt 정리
```powershell
.\scripts\run.ps1 organize
```
동작:
- repo 루트의 `*.txt`를 `data/raw/inbox/`로 안전 이동
- `data/raw/organized/<room>/<yyyy-mm>/`에 hardlink(실패 시 copy) 정리
- `data/raw/manifest.json`에 원본→이동 경로/해시/메타 기록

dry-run:
```powershell
.\scripts\run.ps1 organize --dry
```

### 2) 전처리
```powershell
.\scripts\run.ps1 preprocess
```
출력:
- `data/processed/tokenizer.json`
- `data/processed/train.bin`
- `data/processed/val.bin`
- `data/processed/stats.json`

### 3) 학습/재개학습
```powershell
.\scripts\run.ps1 train
```
동작:
- `checkpoints/<run_name>/latest.pt`가 있으면 자동 resume
- 없으면 새 학습 시작
- `latest.pt`, `best.pt`, `snapshots/step_xxxxxx.pt` 저장
- eval/save 주기마다 샘플 생성 로그 저장: `checkpoints/<run_name>/logs/sample_step_xxxxxx.txt`

중단:
- `Ctrl+C` -> 최신 상태 저장 후 종료
- `checkpoints/<run_name>/STOP` 파일 생성 -> 감지 후 안전 종료

### 4) 즉시 추론/테스트
단발 응답:
```powershell
.\scripts\run.ps1 reply "저녁 뭐먹지"
```

터미널 채팅:
```powershell
.\scripts\run.ps1 chat
```

### 5) 카톡 브리지
기본은 dry-run 권장:
```powershell
.\scripts\run.ps1 bridge --dry
```

실전 전송:
```powershell
.\scripts\run.ps1 bridge --send
```

좌표 보정:
```powershell
python -m chatbot.kakao_bridge --print_mouse --bot_speaker 최근용
```

## 설정 파일
수정 대상:
- `configs/paths.yaml`
- `configs/train.yaml`
- `configs/gen.yaml`

실행 시 현재 설정 스냅샷:
- `checkpoints/<run_name>/config_used.yaml`

환경변수 오버라이드(.env):
- 형식: `CHATBOT_CFG__<config>__<path>__<key>=value`
- 예: `CHATBOT_CFG__train__optimization__learning_rate=0.0002`

## 재개 시 하이퍼파라미터 변경 정책
재개 불가(변경 시 에러):
- 모델 아키텍처/보캐브 관련 (`block_size`, `n_layer`, `n_head`, `n_embd`, `bias`, `dropout`, `vocab_size`)
- 토크나이저 내용(sha256) 변경

재개 가능(변경 시 override 반영 + 로그):
- learning rate, weight decay, batch size, grad accumulation
- eval/save/log interval, max_steps, device/dtype 등 운영 파라미터

override 로그:
- `checkpoints/<run_name>/logs/override_log.json`

## 모델 publish (GitHub 업로드용)
권장: 아티팩트는 1개만 유지

plain:
```powershell
.\scripts\publish_model.ps1
```

encrypted:
```powershell
.\scripts\publish_model.ps1 -Encrypt
```

결과:
- `artifacts/model_latest.pt` 또는 `artifacts/model_latest.enc`
- `artifacts/model_info.json`

## 모델 암호화/복호화 (옵션)
```powershell
python scripts\encrypt_model.py --source checkpoints\room_v1\best.pt --target artifacts\model_latest.enc
python scripts\decrypt_model.py --source artifacts\model_latest.enc --target artifacts\model_latest.pt
```

## 보안 게이트
추론/브리지는 `.env`에 비밀번호가 없으면 실행 거부:
- `CHATBOT_PASSWORD` 또는 `CHATBOT_PASSWORD_HASH`

주의:
- 완전한 보안은 아님. 로컬 코드 수정으로 우회 가능.
- 보호 목적은 무단 사용 난이도 증가(운영 통제 보조)입니다.

## Git/LFS 가이드
기본 정책:
- 원본 txt / processed / checkpoints는 `.gitignore`로 차단
- `artifacts/model_latest.*` 1개만 선택 추적 가능

대용량 모델은 Git LFS 권장:
```powershell
git lfs install
git lfs track "artifacts/model_latest.pt"
git lfs track "artifacts/model_latest.enc"
```

## 자주 겪는 문제
1. `torch.cuda.is_available() == False`
- CUDA용 torch 재설치 필요
- 드라이버/파이썬 버전 호환 확인

2. `Inference/bridge access blocked`
- `.env`에 `CHATBOT_PASSWORD`(또는 hash) 설정 확인

3. `No checkpoint found`
- `configs/gen.yaml`의 checkpoint 우선순위 확인
- `.\scripts\run.ps1 train`으로 latest/best 생성 확인

4. `Incompatible resume`
- 모델 아키텍처/토크나이저 변경됨
- 새 run_name으로 새 학습 시작 권장

## 운영 문서
- 작업 인수인계: `PORTABLE_STATE.txt`
- 다음 Codex 세션 가이드: `AGENTS.md`
