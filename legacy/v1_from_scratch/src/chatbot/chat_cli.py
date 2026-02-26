from __future__ import annotations

import argparse

from .config import load_gen_config
from .console import safe_print
from .inference import InferenceEngine
from .security import require_password


def choose_default_user_speaker(candidates: list[str], bot_speaker: str) -> str:
    for speaker in candidates:
        if speaker != bot_speaker:
            return speaker
    return bot_speaker


def print_help(named_mode: bool) -> None:
    safe_print('/help               - show commands')
    if named_mode:
        safe_print('/speakers           - list speakers')
        safe_print('/bot <name>         - change bot speaker')
        safe_print('/user <name>        - change user speaker')
    safe_print('/temp <value>       - change temperature')
    safe_print('/top_p <value>      - change top_p')
    safe_print('/top_k <value>      - change top_k')
    safe_print('/max_new <value>    - change max_new_tokens')
    safe_print('/reset              - clear chat history')
    safe_print('/exit               - exit')


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive chat CLI for Kakao room model.')
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--config_gen', default='configs/gen.yaml')
    parser.add_argument('--env_path', default='.env')
    parser.add_argument('--password', default='')
    parser.add_argument('--run_name', default='')

    parser.add_argument('--device', default='', choices=['', 'auto', 'cpu', 'cuda'])
    parser.add_argument('--dtype', default='', choices=['', 'auto', 'fp32', 'fp16', 'bf16'])
    parser.add_argument('--bot_speaker', default='')
    parser.add_argument('--user_speaker', default='')
    parser.add_argument('--temperature', type=float, default=-1.0)
    parser.add_argument('--top_p', type=float, default=-1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=-1)
    parser.add_argument('--repetition_penalty', type=float, default=-1.0)
    parser.add_argument('--max_history_turns', type=int, default=-1)
    parser.add_argument('--max_context_tokens', type=int, default=-1)
    args = parser.parse_args()

    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
    sampling_cfg = dict(gen_cfg.get('sampling', {}))
    chat_cfg = dict(gen_cfg.get('chat', {}))
    runtime_cfg = dict(gen_cfg.get('runtime', {}))
    security_cfg = dict(gen_cfg.get('security', {}))
    dialogue_cfg = dict(gen_cfg.get('dialogue', {}))

    require_password(
        security_cfg=security_cfg,
        password=(args.password or None),
        env_path=args.env_path,
    )

    device = args.device or str(runtime_cfg.get('device', 'auto'))
    dtype = args.dtype or str(runtime_cfg.get('dtype', 'auto'))

    engine = InferenceEngine.load(
        checkpoint_path=(args.ckpt or None),
        device=device,
        dtype=dtype,
        gen_config_path=args.config_gen,
        env_path=args.env_path,
        run_name_override=args.run_name,
    )
    speakers = engine.tokenizer.speaker_names
    if not speakers:
        raise RuntimeError('No speaker information in tokenizer.')

    dialogue_mode = engine.dialogue_mode
    named_mode = dialogue_mode == 'named'

    cfg_bot = args.bot_speaker or str(chat_cfg.get('bot_speaker', ''))
    cfg_user = args.user_speaker or str(chat_cfg.get('user_speaker', ''))

    if named_mode:
        bot_speaker = cfg_bot or speakers[0]
        if bot_speaker not in speakers:
            raise ValueError(f'Unknown bot speaker: {bot_speaker}')
        user_speaker = cfg_user or choose_default_user_speaker(speakers, bot_speaker)
        if user_speaker not in speakers:
            raise ValueError(f'Unknown user speaker: {user_speaker}')
    else:
        user_speaker, bot_speaker = engine.resolve_dialogue_speakers(
            user_speaker=(cfg_user or None),
            bot_speaker=(cfg_bot or None),
        )

    temperature = args.temperature if args.temperature >= 0 else float(sampling_cfg.get('temperature', 0.9))
    top_p = args.top_p if args.top_p >= 0 else float(sampling_cfg.get('top_p', 0.95))
    top_k = args.top_k if args.top_k >= 0 else int(sampling_cfg.get('top_k', 120))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else int(sampling_cfg.get('max_new_tokens', 120))
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty >= 0
        else float(sampling_cfg.get('repetition_penalty', 1.02))
    )
    max_turns = args.max_history_turns if args.max_history_turns > 0 else int(dialogue_cfg.get('max_turns', 12))
    max_context_tokens = (
        args.max_context_tokens
        if args.max_context_tokens > 0
        else int(dialogue_cfg.get('max_context_tokens', engine.model.config.block_size))
    )
    history: list[tuple[str, str]] = []

    safe_print(f'[ready] checkpoint={engine.checkpoint_path}')
    safe_print(f'[ready] mode={dialogue_mode} turns={max_turns} context_tokens={max_context_tokens}')
    safe_print('Type /help for commands.')

    while True:
        try:
            user_input = input('> ').strip()
        except (KeyboardInterrupt, EOFError):
            safe_print('\nbye')
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            cmd, *rest = user_input.split(' ', 1)
            arg = rest[0].strip() if rest else ''

            if cmd == '/help':
                print_help(named_mode=named_mode)
            elif cmd == '/speakers':
                if named_mode:
                    safe_print(', '.join(speakers))
                else:
                    safe_print('not available in anonymous mode')
            elif cmd == '/bot':
                if not named_mode:
                    safe_print('not available in anonymous mode')
                    continue
                if arg not in speakers:
                    safe_print(f'unknown speaker: {arg}')
                    continue
                bot_speaker = arg
                safe_print('bot updated')
            elif cmd == '/user':
                if not named_mode:
                    safe_print('not available in anonymous mode')
                    continue
                if arg not in speakers:
                    safe_print(f'unknown speaker: {arg}')
                    continue
                user_speaker = arg
                safe_print('user updated')
            elif cmd == '/temp':
                temperature = float(arg)
                safe_print(f'temperature={temperature}')
            elif cmd == '/top_p':
                top_p = float(arg)
                safe_print(f'top_p={top_p}')
            elif cmd == '/top_k':
                top_k = int(arg)
                safe_print(f'top_k={top_k}')
            elif cmd == '/max_new':
                max_new_tokens = int(arg)
                safe_print(f'max_new_tokens={max_new_tokens}')
            elif cmd == '/reset':
                history = []
                safe_print('history cleared')
            elif cmd == '/exit':
                safe_print('bye')
                break
            else:
                safe_print('unknown command. type /help')
            continue

        history.append(('user', user_input) if not named_mode else (user_speaker, user_input))

        if engine.debug_return_raw:
            reply, raw = engine.generate_reply_with_raw(
                history=history,
                user_speaker=user_speaker,
                bot_speaker=bot_speaker,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_turns_override=max_turns,
                max_context_tokens_override=max_context_tokens,
            )
            safe_print(reply)
            safe_print(f'[raw] {raw}')
        else:
            reply = engine.generate_reply(
                history=history,
                user_speaker=user_speaker,
                bot_speaker=bot_speaker,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_turns_override=max_turns,
                max_context_tokens_override=max_context_tokens,
            )
            safe_print(reply)

        history.append(('bot', reply) if not named_mode else (bot_speaker, reply))


if __name__ == '__main__':
    main()
