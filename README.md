```
 █████╗ ██╗   ██╗ ██████╗  ██████╗  █████╗ ██████╗  ██████╗
██╔══██╗██║   ██║██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗██╔═══██╗
███████║██║   ██║██║   ██║██║      ███████║██║  ██║██║   ██║
██╔══██║╚██╗ ██╔╝██║   ██║██║      ██╔══██║██║  ██║██║   ██║
██║  ██║ ╚████╔╝ ╚██████╔╝╚██████╗ ██║  ██║██████╔╝╚█████╔╝
╚═╝  ╚═╝  ╚═══╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝   ╚═════╝
                               C L I   A S S I S T A N T
```

# Avocado AI CLI

Maximum light agent for economical AI: local resources or cloud tokens are spent only where it is really needed.

## What is Avocado AI CLI

**Avocado** is a lightweight terminal AI agent for coding, analysis, and file work. It is designed for people who want to **distribute tasks between models explicitly** instead of dumping everything into one prompt and overloading expensive heavy models.

- A cheap or local model handles routine work.
- A smart model (expert) gets only precise subtasks with minimal tokens and no extra history — it can be used pointwise, whether it is free or an expensive heavy model.
- Session history is compressed by AI into compact long-term memory (`memory.md`).
- Tools are connected manually; the system prompt is minimal (~50 tokens).
- Works everywhere Python runs, including **Termux on Android**.

## Expert models — one of the key features

Avocado lets you attach different expert models for different domains instead of stuffing everything into a single model. The main model delegates only the exact subtask to the chosen expert.

Examples of expert models by domain:

| Domain | Example expert models | What they do |
|---|---|---|
| Coding | Qwen2.5-Coder, Qwen3-Coder, DeepSeek-Coder | Write, review, and fix code |
| Math and logic | DeepSeek-Math, Qwen3 (math reasoning) | Solve formulas, algorithms, logic puzzles |
| Medicine | Meditron, BioMistral, fine-tuned Llama/Mistral | Medical consultation, symptom analysis |
| Translation | NLLB, M2M100, Qwen (multilingual) | Translate between languages |
| Space industry / other domain | Domain fine-tunes on Llama / Mistral / Qwen | Technical docs, engineering calculations |

Each expert is **stateless**: it does not see the whole conversation, session plan, or memory. It receives only what the main model explicitly passes, so even an expensive heavy model is used only for a narrow part of the task — you do not pay for it on every step.

## Why memory compression matters

When a model fills more than half of its context window, it often starts to hallucinate and get confused, even if it formally still fits. AI memory compression in Avocado lets models with a small context window not only fit into it but also stay in the zone of clarity — the session history is reduced to the most important facts, and the agent never reaches the level where confusion begins.

## Advantages

| Mechanism | What it gives |
|---|---|
| Expert delegation | Expensive/heavy model works only on precise subtasks, no junk context |
| AI memory compression | Whole step journal becomes compact `memory.md`; long sessions do not burn tokens |
| Manual tool connection | Only connected tools appear in the prompt; less noise, lower cost |
| Minimal system prompt | About 50 tokens instead of a huge tool schema |
| Portability | Python CLI, minimal dependencies, runs in Termux, on old laptops, Raspberry Pi |

## Memory: what is compressed and what is not

| Layer | File | Compressed? | In context? | Purpose |
|---|---|---|---|---|
| Plan / status | `plan.json` | No | Yes | Current task, steps, status |
| Long-term memory | `memory.md` | Yes (AI compression) | Yes | Facts, data, open questions, fixes |
| Raw journal of current run | `current_run.jsonl` | No | No (compression only) | Raw steps, tool calls, errors |
| Archived journal | `archive/run-*.jsonl` | No | No | Audit, debug, recovery |
| Compression metadata | `memory.meta.json` | No | No | Token statistics before/after |

The compressor compresses the previous compressed summary plus new steps from the current run. It **does not** read the whole accumulated archive.

## Quick start

```bash
pip install -e .
avocado --help
```

Keys via environment variables:

```bash
export GROQ_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

First run:

```bash
avocado --model Groq/qwen/qwen3.8-27b
```

## Usage

Slash commands inside CLI:

| Command | Purpose |
|---|---|
| `/help` | help |
| `/sessions` | list sessions |
| `/new [name]` | create new session |
| `/switch <id\|name>` | compress current and switch to another |
| `/plan` | show plan |
| `/memory` | memory paths and preview |
| `/tools ...` | manage tools |
| `/experts ...` | manage experts |
| `/compress` | compress journal into memory |
| `/context` | estimate context usage |
| `/reset [--hard]` | clear history; `--hard` also clears memory |
| `/quit`, `/exit` | save and quit |

### Termux on Android

```bash
pkg update && pkg install python
cd Avocado_AI_CLI
pip install -e .
export GROQ_API_KEY="..."
avocado --model Groq/qwen/qwen3.8-27b
```

### Experts: precise delegation

An expert is not a second agent with full history. It receives only what the main model explicitly passes and does not see the whole conversation, plan, or memory. By default an expert has no tools; tools are enabled explicitly.

```text
/experts enable logic_specialist
/experts tools code_reviewer read_file
```

## Architecture

```text
User --> avocado CLI --> SessionManager --> Engine --> Model (OpenAI-compatible API)
                             |             |
                             |             --> Tools (manual)
                             |             --> Experts (stateless)
                             --> MemoryCompressor --> Model
Session: plan.json, memory.md, current_run.jsonl
```

Key session files:

```text
session.json        metadata
plan.json           current plan (not compressed)
memory.md           compressed long-term memory
memory.meta.json    compression statistics
current_run.jsonl   raw step journal
archive/            archived journals
```

Flow of one request:

1. Gather instructions: system prompt + plan + memory + expert hints.
2. Choose engine by model capabilities: code loop → tool calling → chat.
3. Execute task; steps are written to the journal.
4. `/compress` turns the journal into updated `memory.md`.
5. Next run starts from compact context, not the whole history.

## Configuration

Config: `~/.config/avocado/config.json` (or `$AVOCADO_HOME/config.json`). Sessions: `~/.local/share/avocado/sessions`.

Example:

```json
{
  "agent_config": {
    "agent_name": "Avocado",
    "agent_type": "code",
    "max_steps": 20,
    "memory": {
      "compression": {
        "model": "",
        "max_summary_tokens": 0,
        "max_transcript_tokens": 0
      }
    }
  },
  "settings": {
    "providers": [
      {
        "id": "groq",
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "models": [
          {
            "name": "qwen/qwen3.8-27b",
            "supports_code_loop": true,
            "supports_tool_calling": false,
            "params": { "max_tokens": 4096 }
          }
        ]
      }
    ],
    "experts": [
      {
        "role": "logic_specialist",
        "name": "Logic Specialist",
        "model": "Groq/openai/gpt-oss-120b",
        "agent_type": "chat",
        "allow_tools": false,
        "tools": []
      }
    ]
  }
}
```

`max_summary_tokens: 0` and `max_transcript_tokens: 0` disable hard compression limits.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

116 tests, no real API keys required (models are replaced with deterministic fakes).

## Security

- API keys are taken only from environment variables or OS keyring; none are stored in the repository.
- Tools are connected manually; `run_shell` / `bash` are inactive until explicitly added.
- `delete_path` refuses to delete protected system paths.
- Sessions, plan, and memory are stored locally.
- No telemetry or analytics.

Code loop executes Python locally — powerful but not a full sandbox. Run the agent with the same restrictions you would use for any local script.

## Limitations

- Code execution is local, without a container sandbox.
- Compression quality depends on the compression model.
- No cloud session synchronization.
- Terminal interface, no GUI.
- Providers connect through an OpenAI-compatible layer, not specialized SDKs.