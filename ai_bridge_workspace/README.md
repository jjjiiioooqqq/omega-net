# AI Bridge Workspace

A workspace for connecting Claude (or any operator of this repo) to external
AI models over their public APIs:

- **ChatGPT** (OpenAI) — default model `sol-ultra` (change it if your account
  uses a different model ID; any OpenAI chat model works)
- **Google AI Studio** (Gemini API) — default model `gemini-2.5-pro`

Both providers are reached through the OpenAI-compatible Chat Completions
protocol, so a single stdlib-only script (`bridge.py`, no pip installs)
handles both.

## Setup

Copy the env template and fill in the key(s) you have:

```bash
cp .env.example .env
# edit .env, then:
set -a; source .env; set +a
```

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | key for the `chatgpt` provider (platform.openai.com) |
| `GEMINI_API_KEY` | key for the `google` provider (aistudio.google.com) |
| `SOL_ULTRA_MODEL` | override the ChatGPT model ID (default `sol-ultra`) |
| `GOOGLE_MODEL` | override the Gemini model ID (default `gemini-2.5-pro`) |
| `BRIDGE_SYSTEM` | override the system prompt sent to either model |

## Talking to the models

```bash
# interactive chat with ChatGPT
python ai_bridge_workspace/bridge.py chat

# interactive chat with Google AI Studio
python ai_bridge_workspace/bridge.py --provider google chat

# one-shot question
python ai_bridge_workspace/bridge.py send "Summarize the engint package design"

# pick a specific model
python ai_bridge_workspace/bridge.py --provider google --model gemini-2.5-flash chat
```

Every session is saved to `transcripts/<provider>_<timestamp>.json`
(gitignored). Resume one with:

```bash
python ai_bridge_workspace/bridge.py chat --resume ai_bridge_workspace/transcripts/<file>.json
```

## Notes

- Transcripts and `.env` are gitignored — keys and conversation logs never
  get committed.
- If a model ID is rejected (e.g. `sol-ultra` isn't available on your
  account), the API error is printed verbatim; set `SOL_ULTRA_MODEL` /
  `GOOGLE_MODEL` or pass `--model` with a valid ID.
- `--base` lets you point the same script at any other OpenAI-compatible
  endpoint (local servers, proxies, other vendors).
