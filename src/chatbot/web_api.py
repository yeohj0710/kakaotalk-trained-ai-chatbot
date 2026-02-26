from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .security import require_password
from .sft_config import load_sft_config
from .sft_infer import SFTInferenceEngine, configure_console_io


class ChatTurn(BaseModel):
    role: Literal["user", "bot"]
    text: str = Field(min_length=1, max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    adapter: str


@dataclass
class AppState:
    config_sft: str
    env_path: str
    adapter_path: str
    run_name: str
    cfg: dict[str, Any]
    engine: SFTInferenceEngine | None
    engine_lock: threading.Lock
    generation_lock: threading.Lock


def create_app(
    config_sft: str = "configs/sft.yaml",
    env_path: str = ".env",
    adapter_path: str = "",
    run_name: str = "",
) -> FastAPI:
    cfg = load_sft_config(config_path=config_sft, env_path=env_path)
    state = AppState(
        config_sft=config_sft,
        env_path=env_path,
        adapter_path=adapter_path,
        run_name=run_name,
        cfg=cfg,
        engine=None,
        engine_lock=threading.Lock(),
        generation_lock=threading.Lock(),
    )

    app = FastAPI(title="KakaoTalk Chatbot Local API", version="0.1.0")
    app.state._chatbot = state

    def get_state() -> AppState:
        return app.state._chatbot  # type: ignore[attr-defined]

    def ensure_auth(password: str | None) -> None:
        security_cfg = dict(get_state().cfg.get("security", {}))
        try:
            require_password(security_cfg=security_cfg, password=password, env_path=get_state().env_path)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def get_engine() -> SFTInferenceEngine:
        st = get_state()
        with st.engine_lock:
            if st.engine is None:
                st.engine = SFTInferenceEngine.load(
                    config_sft=st.config_sft,
                    env_path=st.env_path,
                    adapter_path=st.adapter_path,
                    run_name_override=st.run_name,
                )
            return st.engine

    @app.get("/health")
    def health() -> dict[str, str]:
        st = get_state()
        if st.engine is None:
            return {"status": "ok", "engine": "cold"}
        return {"status": "ok", "engine": "ready", "adapter": str(st.engine.adapter_dir)}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        x_chatbot_password: str | None = Header(default=None),
    ) -> ChatResponse:
        ensure_auth(x_chatbot_password)
        try:
            engine = get_engine()
            history = [(item.role, item.text) for item in payload.history if item.text.strip()]
            with get_state().generation_lock:
                reply = engine.reply(history=history, user_text=payload.message.strip())
            return ChatResponse(reply=reply, adapter=str(engine.adapter_dir))
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return app


def main() -> None:
    configure_console_io()
    parser = argparse.ArgumentParser(description="Run local HTTP API server for chatbot inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--run_name", default="")
    args = parser.parse_args()

    app = create_app(
        config_sft=args.config_sft,
        env_path=args.env_path,
        adapter_path=args.adapter,
        run_name=args.run_name,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
