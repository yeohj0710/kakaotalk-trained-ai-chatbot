# RUNBOOK_TOMORROW_2026-02-26 (FULL STEP-BY-STEP)

이 문서는 "내일 출근해서 처음부터 다시 켜는 사람" 기준으로 작성했다.  
그대로 복붙하면 된다.

---

## 0) 전체 구조 먼저 이해 (1분)

프로젝트는 2개다.

1. 모델/학습/API 서버:
- `C:\dev\kakaotalk-trained-ai-chatbot`

2. 웹 UI (Next.js):
- `C:\dev\kakaotalk-chat-web`

흐름:
- 학습은 모델 repo에서 계속 진행
- 모델 API 서버가 `127.0.0.1:8000`에서 응답
- 웹 UI가 모델 API를 호출
- 외부 접속은 cloudflared 터널로 `8000`을 공개
- Vercel은 그 터널 URL을 사용해서 모델과 통신

---

## 1) 오늘 반영된 핵심 수정사항 (요약)

1. 모델 API 서버 추가
- `src/chatbot/web_api.py`
- `GET /health`, `POST /v1/chat`

2. CLI에 API 서버 명령 추가
- `python -m chatbot.sft_ops serve`

3. 최신 체크포인트 자동 실행 스크립트
- `scripts/serve_latest.sh`
- run_name이 없으면 `_cpt` 자동 fallback

4. 웹 API 프록시 추가
- `C:\dev\kakaotalk-chat-web\src\app\api\chat\route.ts`

5. 웹 채팅 UI 추가
- `C:\dev\kakaotalk-chat-web\src\app\page.tsx`

6. 터널 자동 스크립트 추가
- `C:\dev\kakaotalk-chat-web\scripts\open_model_tunnel.sh`
- `npm run tunnel:model`

7. env 따옴표 방어 로직
- `MODEL_API_PASSWORD="0903"` 같은 형태여도 route에서 trim 처리
- 그래도 권장값은 따옴표 없이 입력

---

## 2) 내일 시작 전에 준비물 체크 (2분)

Git Bash 4개 띄워서 아래 이름으로 사용:

- 터미널 A: 학습(train)
- 터미널 B: 모델 API 서버(serve)
- 터미널 C: 웹 로컬(dev)
- 터미널 D: 터널(cloudflared)

### 2-1) 경로 존재 확인
```bash
ls /c/dev/kakaotalk-trained-ai-chatbot
ls /c/dev/kakaotalk-chat-web
```

### 2-2) 모델 repo venv 확인
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
ls .venv/Scripts/python.exe
```

### 2-3) cloudflared 설치 확인
```bash
cloudflared --version
```
안 나오면(명령어 없음):
```powershell
winget install -e --id Cloudflare.cloudflared --accept-package-agreements --accept-source-agreements
```
설치 후 Git Bash를 **새로** 열어야 인식될 수 있다.

---

## 3) 터미널 A - 학습 이어서 실행 (가장 먼저)

> 이미 학습중이라면 이 단계는 건너뛰고 유지하면 된다.

```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
source .venv/Scripts/activate
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

정상 포인트:
- 이전 checkpoint에서 resume
- step 숫자가 계속 증가
- `save_steps` 주기마다 checkpoint 저장

멈추는 법:
- `Ctrl+C` (안전 저장 후 종료)

다시 이어서:
- 동일 명령 재실행하면 자동 resume

---

## 4) 터미널 B - 모델 API 서버 실행 (latest checkpoint 기준)

```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
bash scripts/serve_latest.sh
```

정상 로그 예시:
- `[serve] run=...`
- `[serve] adapter=...checkpoint-xxxx`
- `Uvicorn running on http://127.0.0.1:8000`

중요:
- 스크립트가 `room_lora_qwen25_3b_base`를 못 찾으면
  자동으로 `room_lora_qwen25_3b_base_cpt`를 사용하도록 수정됨.

### 4-1) health 확인 (새 터미널에서)
```bash
curl http://127.0.0.1:8000/health
```

정상 응답 예:
```json
{"status":"ok","engine":"cold"}
```
또는
```json
{"status":"ok","engine":"ready","adapter":"..."}
```

---

## 5) 터미널 C - 웹 로컬 실행

```bash
cd /c/dev/kakaotalk-chat-web
npm install
npm run dev
```

브라우저:
- `http://localhost:3000`

### 5-1) 웹 env 설정 (최초 1회 또는 값 변경 시)

`C:\dev\kakaotalk-chat-web\.env.local` 파일 내용:
```env
MODEL_API_URL=http://127.0.0.1:8000/v1/chat
MODEL_API_PASSWORD=여기에_모델_repo_.env의_CHATBOT_PASSWORD_값
```

주의:
- 값 앞뒤 공백 금지
- 권장: 따옴표 없이 입력

---

## 6) 터미널 D - 터널 실행 (외부/Vercel 연결용)

```bash
cd /c/dev/kakaotalk-chat-web
npm run tunnel:model
```

정상 시 콘솔에 아래가 출력됨:
```text
MODEL_API_URL=https://xxxx.trycloudflare.com/v1/chat
```

이 값을 Vercel에 넣으면 된다.

중요:
- 터널 터미널을 끄면 URL이 죽는다.
- 다시 켜면 URL이 바뀔 수 있다.

---

## 7) Vercel 설정 (외부 기기 접속용)

Vercel 프로젝트 Settings > Environment Variables:

1. `MODEL_API_URL`
- 값: 터미널 D 출력값 (`https://...trycloudflare.com/v1/chat`)

2. `MODEL_API_PASSWORD`
- 값: 모델 repo `.env`의 `CHATBOT_PASSWORD`와 동일

저장 후 redeploy.

---

## 8) 내일 실제 운영 순서 (요약판)

순서 꼭 지키기:

1. 터미널 A에서 train 실행(또는 기존 학습 유지)
2. 터미널 B에서 `bash scripts/serve_latest.sh`
3. health 체크 `curl http://127.0.0.1:8000/health`
4. 터미널 C에서 `npm run dev` (로컬 테스트)
5. 터미널 D에서 `npm run tunnel:model` (외부 연동)
6. 터널 URL을 Vercel `MODEL_API_URL`에 반영

---

## 9) 자주 나는 오류 + 즉시 해결

### 오류 1) `no adapter found for run: room_lora_qwen25_3b_base`
원인:
- 기본 run_name 폴더가 없고 `_cpt`만 있는 상태

해결:
- 이미 스크립트에 fallback 반영됨
- 그냥 다시:
```bash
bash scripts/serve_latest.sh
```
- 강제 지정이 필요하면:
```bash
RUN_NAME=room_lora_qwen25_3b_base_cpt bash scripts/serve_latest.sh
```

### 오류 2) `WinError 10048` (포트 8000 사용 중)
원인:
- 기존 API 서버 프로세스가 이미 떠 있음

PowerShell에서:
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
Stop-Process -Id <PID> -Force
```
그 후 터미널 B 명령 재실행.

### 오류 3) 401 Unauthorized
원인:
- 비밀번호 불일치 (`MODEL_API_PASSWORD` != `CHATBOT_PASSWORD`)

해결:
- 모델 repo `.env`의 `CHATBOT_PASSWORD` 확인
- 웹 `.env.local` + Vercel `MODEL_API_PASSWORD`를 동일값으로 재설정

### 오류 4) cloudflared 경고 로그(cert 관련)
원인:
- Quick Tunnel에서 흔한 경고

해결:
- URL이 발급되고 연결 로그가 뜨면 대체로 정상
- 핵심은 `MODEL_API_URL=https://.../v1/chat` 출력 여부

### 오류 5) 웹은 열리는데 답변이 안 옴(5xx)
체크 순서:
1. `curl http://127.0.0.1:8000/health`
2. 터미널 B(API) 살아있는지 확인
3. 비밀번호 일치 확인
4. Vercel env 재배포 확인

---

## 10) 보안 주의

- 스크린샷에 비밀번호가 노출되었을 수 있다.
- 내일 첫 작업으로 비밀번호 교체 권장:
  - 모델 repo `.env`의 `CHATBOT_PASSWORD` 변경
  - 웹 `.env.local` 및 Vercel `MODEL_API_PASSWORD`도 동일하게 변경

---

## 11) 종료할 때 체크리스트

퇴근 전 확인:

1. 학습 계속 돌릴지 결정
- 계속 돌릴 거면 터미널 A 유지
- 중단할 거면 `Ctrl+C`로 정상 종료

2. 터널은 필요 없으면 종료
- 터널 URL은 매번 바뀌어도 정상

3. 다음날 바로 보기 위한 파일
- 이 파일: `RUNBOOK_TOMORROW_2026-02-26.md`
- 상태 개요: `PORTABLE_STATE.txt`

---

## 12) 복붙용 "내일 시작 세트"

### (A) 모델 repo: 학습 + API
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
source .venv/Scripts/activate
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

새 터미널:
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
bash scripts/serve_latest.sh
```

### (B) 웹 repo: 로컬 + 터널
```bash
cd /c/dev/kakaotalk-chat-web
npm install
npm run dev
```

새 터미널:
```bash
cd /c/dev/kakaotalk-chat-web
npm run tunnel:model
```

