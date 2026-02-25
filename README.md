# KakaoTalk Room Chatbot (From Scratch)

카카오톡 내보내기 `.txt` 원본만으로, 사전학습 모델 없이 GPT 계열 모델을 처음부터 학습해
톡방 말투를 흉내내는 로컬 챗봇을 만드는 프로젝트입니다.

## 구성
- `chatbot.preprocess`: raw txt 파싱 + 정제 + 토크나이저/학습 바이너리 생성
- `chatbot.train`: character/byte-level GPT 학습 (PyTorch)
- `chatbot.chat_cli`: 터미널 대화 테스트
- `chatbot.generate`: 단발 생성/스크립트용 호출
- `chatbot.kakao_bridge`: 카카오톡 PC 창과 연결하는 UI 자동화 브리지 (기본 dry-run)

## 1) 환경 준비
```powershell
cd c:\dev\kakaotalk-trained-ai-chatbot
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()`가 `True`여야 GPU 학습됩니다.

## 2) 데이터 전처리
현재 폴더의 `*.txt`를 읽어서 학습 데이터 생성:
```powershell
python -m chatbot.preprocess --input_glob "*.txt" --output_dir data/processed --val_ratio 0.02 --mask_urls
```

생성물:
- `data/processed/tokenizer.json`
- `data/processed/train.bin`
- `data/processed/val.bin`
- `data/processed/messages.jsonl`
- `data/processed/stats.json`

## 3) 모델 학습
기본 학습(16GB GPU 권장):
```powershell
python -m chatbot.train `
  --data_dir data/processed `
  --out_dir checkpoints/base `
  --block_size 256 `
  --n_layer 8 --n_head 8 --n_embd 512 `
  --batch_size 16 `
  --grad_accum_steps 1 `
  --max_steps 20000 `
  --eval_interval 200 `
  --save_interval 400 `
  --device auto --dtype auto
```

체크포인트:
- `checkpoints/base/latest.pt`
- `checkpoints/base/best.pt`
- `checkpoints/base/step_XXXXXX.pt`

재개학습:
```powershell
python -m chatbot.train --data_dir data/processed --out_dir checkpoints/base --resume checkpoints/base/latest.pt
```

## 4) 터미널 채팅 테스트
```powershell
python -m chatbot.chat_cli --ckpt checkpoints/base/best.pt --bot_speaker "원하는화자명"
```

CLI 명령어:
- `/speakers` 학습된 화자 목록
- `/bot 이름` 봇 화자 변경
- `/user 이름` 사용자 화자 변경
- `/temp`, `/top_p`, `/top_k`, `/max_new` 생성 파라미터 조정
- `/reset`, `/exit`

## 5) 카카오톡 브리지 (주의)
공식 카카오톡 Bot API가 없어서, PC 앱 UI 자동화 방식입니다.
좌표가 틀리면 오작동할 수 있으므로 처음엔 반드시 dry-run으로 테스트하세요.

좌표 캘리브레이션:
```powershell
python -m chatbot.kakao_bridge --ckpt checkpoints/base/best.pt --bot_speaker "원하는화자명" --chat_xy 0,0 --input_xy 0,0 --print_mouse
```

```powershell
python -m chatbot.kakao_bridge `
  --ckpt checkpoints/base/best.pt `
  --bot_speaker "원하는화자명" `
  --window_title "카카오톡" `
  --chat_xy 500,320 `
  --input_xy 520,980
```

실제 전송:
```powershell
python -m chatbot.kakao_bridge ... --send
```

## 권장 학습 전략
- 1차: `n_layer=6, n_head=6, n_embd=384, max_steps=5000`으로 빠른 검증
- 2차: 품질 확인 후 `n_layer=8~10, n_embd=512~640, max_steps=20k~80k`
- 과적합 방지: `dropout 0.1~0.2`, 중복 제거 유지

## 보안/개인정보
- `.gitignore`에 `*.txt`가 포함되어 원본 대화 파일은 기본적으로 커밋되지 않습니다.
- 체크포인트도 대화 데이터 패턴을 내재하므로 외부 공유 전 주의하세요.
