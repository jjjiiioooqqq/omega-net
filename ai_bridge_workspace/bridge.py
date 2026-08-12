#!/usr/bin/env python3
"""Chat bridge for talking to external AI models: ChatGPT and Google AI Studio.

Stdlib-only client for OpenAI-compatible Chat Completions endpoints (Google
AI Studio's Gemini API exposes one too, so both providers share a single
code path). Supports an interactive REPL and one-shot messages, and records
every exchange as a JSON transcript so conversations can be resumed or
audited later.

Usage:
    export OPENAI_API_KEY=sk-...            # for --provider chatgpt
    export GEMINI_API_KEY=...               # for --provider google
    python bridge.py chat                                # ChatGPT, interactive
    python bridge.py --provider google chat              # Gemini, interactive
    python bridge.py send "Hello over there"             # one-shot message
    python bridge.py chat --resume transcripts/<session>.json

Configuration (env vars override the preset defaults):
    OPENAI_API_KEY      API key for the chatgpt provider
    GEMINI_API_KEY      API key for the google provider (GOOGLE_API_KEY also works)
    SOL_ULTRA_MODEL     chatgpt model ID   (default: sol-ultra)
    GOOGLE_MODEL        google model ID    (default: gemini-2.5-pro)
    BRIDGE_SYSTEM       system prompt override for either provider
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
TRANSCRIPT_DIR = WORKSPACE / "transcripts"

PROVIDERS = {
    "chatgpt": {
        "base": "https://api.openai.com/v1",
        "key_env": ("OPENAI_API_KEY",),
        "model": os.environ.get("SOL_ULTRA_MODEL", "sol-ultra"),
    },
    "google": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "model": os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro"),
    },
}

DEFAULT_SYSTEM = os.environ.get(
    "BRIDGE_SYSTEM",
    "You are speaking with Claude (an Anthropic model) through a bridge "
    "script in the omega-net repository. Be direct and technical.",
)


class BridgeError(RuntimeError):
    pass


def _api_key(provider: dict) -> str:
    for name in provider["key_env"]:
        key = os.environ.get(name)
        if key:
            return key
    raise BridgeError(
        f"No API key found. Export one of: {', '.join(provider['key_env'])}"
    )


def complete(messages: list[dict], model: str, base: str, key: str,
             timeout: float = 120.0) -> str:
    """Send the conversation to a Chat Completions endpoint, return the reply text."""
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise BridgeError(f"API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BridgeError(f"Network error reaching {base}: {exc.reason}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise BridgeError(f"Unexpected response shape: {json.dumps(payload)[:500]}") from exc


def _new_transcript(provider_name: str) -> Path:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return TRANSCRIPT_DIR / f"{provider_name}_{stamp}.json"


def _save(path: Path, provider_name: str, model: str, messages: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "provider": provider_name,
                "model": model,
                "updated": datetime.now(timezone.utc).isoformat(),
                "messages": messages,
            },
            indent=2,
        )
    )


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text())["messages"]


def cmd_send(args: argparse.Namespace) -> int:
    messages = [{"role": "system", "content": args.system}] if args.system else []
    messages.append({"role": "user", "content": args.message})
    reply = complete(messages, args.model, args.base, args.key)
    messages.append({"role": "assistant", "content": reply})
    _save(_new_transcript(args.provider), args.provider, args.model, messages)
    print(reply)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    if args.resume:
        path = Path(args.resume)
        messages = _load(path)
        print(f"[resumed {path.name}, {len(messages)} prior messages]")
    else:
        path = _new_transcript(args.provider)
        messages = [{"role": "system", "content": args.system}] if args.system else []
    print(f"[provider: {args.provider} | model: {args.model} | endpoint: {args.base}]")
    print("[type your message; 'exit' or Ctrl-D to quit]\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": line})
        try:
            reply = complete(messages, args.model, args.base, args.key)
        except BridgeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        _save(path, args.provider, args.model, messages)
        print(f"\n{args.model}> {reply}\n")
    _save(path, args.provider, args.model, messages)
    print(f"[transcript: {path}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="chatgpt",
                        help="which preset endpoint to talk to (default: chatgpt)")
    parser.add_argument("--model", help="override the provider's default model ID")
    parser.add_argument("--base", help="override the provider's API base URL")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt")
    sub = parser.add_subparsers(dest="command", required=True)

    p_send = sub.add_parser("send", help="send a single message and print the reply")
    p_send.add_argument("message")
    p_send.set_defaults(func=cmd_send)

    p_chat = sub.add_parser("chat", help="interactive chat session")
    p_chat.add_argument("--resume", help="path to a transcript JSON to continue")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    preset = PROVIDERS[args.provider]
    args.model = args.model or preset["model"]
    args.base = (args.base or preset["base"]).rstrip("/")
    try:
        args.key = _api_key(preset)
        return args.func(args)
    except BridgeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
