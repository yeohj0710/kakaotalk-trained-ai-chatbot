# 초간단 실행법

[0] 최초 1회 (설치 + .env 만들기) — Git Bash에서 아래 그대로 실행
cd /c/dev/kakaotalk-trained-ai-chatbot
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
notepad.exe .env

[1] .env에 반드시 넣을 값 (다른 PC에서도 “똑같이” 넣어야 동일하게 사용/이어짐)

- CHATBOT_PASSWORD=내비번 # 추론/채팅/브리지 실행 게이트(문 앞 비번)
- CHATBOT_MODEL_KEY=내암호화키 # .enc(암호화 모델) 쓸 때만 필요(암호화할 때 쓴 키와 동일해야 로딩됨)

[2] 학습(훈련) — 순서대로 실행 (Git Bash에서 PowerShell 스크립트 호출)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 organize
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 preprocess
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 train

- 중단: Ctrl+C
- 재개: 다시 train 실행 (자동 resume)

[3] 추론(테스트)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 reply "오늘 뭐함"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 chat

[4] (옵션) 카톡 브리지
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 bridge --dry
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 bridge --send

[5] 다른 PC에서 이어서 하기(핵심만)

- 추론만: artifacts/model_latest.(pt|enc) 를 새 PC에 복사 + .env에 CHATBOT_PASSWORD(및 enc면 MODEL_KEY) 동일하게
- 학습 이어서: 위 + checkpoints/<run_name>/ 폴더까지 통째로 복사 (latest.pt 포함)

[Output Policy]
- `run.ps1 reply`, `run.ps1 chat`, `run.ps1 bridge` output/send text-only by default (role/speaker prefix hidden).
- If you need raw output for debugging, set `output.text_only: false` or `debug.return_raw: true` in `configs/gen.yaml`.
- Default dialogue mode is now anonymous 2-person chat (ChatGPT-style): no real-name speaker labels are shown in `reply/chat/bridge`.
- `run.ps1 bridge --dry` now runs local test mode (no Kakao window copy/click needed). Use `run.ps1 bridge --dry --ui_dry` when you want UI polling without sending.
