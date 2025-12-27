# Gravity Agents

**Do LLM-based agents use explicit physics laws or rely on trial-and-error when gravity changes?**

This experimental framework tests whether giving AI agents explicit physical formulas (F = m·g) helps them adapt to gravity changes better than vague descriptions ("normal Earth gravity").

## Hypotheses

- **H1 - Law vs Story**: Agents with explicit physical law adapt better than story-based agents
- **H2 - RL vs No-RL**: RL-trained agents adapt more robustly than pure reasoning agents
- **H3 - Information vs Silence**: Agents told about gravity changes adapt better than silent changes

## Project Structure

```
gravity-agents/
├── web-env/                 # Physics environment (three.js + cannon-es)
│   ├── server.js            # HTTP/WebSocket API server
│   └── src/
│       ├── physics/         # Physics engine wrapper
│       └── tasks/           # Gap crossing & throw tasks
│
├── python-orchestrator/     # RL training & evaluation
│   ├── config.py            # Experiment configuration
│   ├── env_client.py        # Environment API client
│   ├── llm_policy.py        # Gemini/Groq policy server
│   ├── atropos_env.py       # Atropos RL wrapper
│   ├── logger.py            # Data logging (Parquet/JSON)
│   └── run_experiment.py    # Main experiment runner
│
└── analysis/                # Results analysis
    └── analyze_results.py   # Plots & hypothesis testing
```

## Quick Start

### 1. Start the Physics Environment

```bash
cd web-env
npm install
node server.js
```

Server runs at http://localhost:3000

### 2. Set Up Python Environment

```bash
cd python-orchestrator
pip install -r requirements.txt
```

### 3. Run Experiments

```bash
# Single agent/task evaluation
python run_experiment.py --agent RL-F --task gap --mode eval --condition explained

# Full experiment matrix
python run_experiment.py --agent all --task all --mode full --episodes 100
```

### 4. Analyze Results

```bash
cd analysis
python analyze_results.py --log-dir ../python-orchestrator/logs
```

## Experimental Conditions

| Agent | Description | Gravity Text |
|-------|-------------|--------------|
| RL-F | RL + Formula | "F = m·g with g = 9.81 m/s²" |
| RL-N | RL + Normal | "normal Earth gravity" |
| NRL-F | No-RL + Formula | Same as RL-F, no training |

| Condition | Gravity | Message |
|-----------|---------|---------|
| Baseline | 9.81 | Same as training |
| Silent | 4.9 | No change in description |
| Explained | 4.9 | Updated gravity description |

## Tasks

### Gap Crossing
Jump across a gap between two platforms. Success = land in goal zone.

**Actions**: forward, back, left, right, jump, idle

### Throw Block
Pick up a block and throw it into an elevated basket.

**Actions**: forward, back, left, right, pick, drop, throw_weak, throw_medium, throw_strong, idle

## API Reference

### Environment Server

**POST /reset**
```json
{
  "sessionId": "agent-1",
  "config": {
    "task": "gap",
    "gravity": 9.81,
    "seed": 42
  }
}
```

**POST /step**
```json
{
  "sessionId": "agent-1",
  "action": "jump"
}
```

**GET /info** - Available tasks and actions

**GET /health** - Server status

## Configuration

API keys in `python-orchestrator/config.py`:
- `GEMINI_API_KEY` - Google Gemini
- `GROQ_API_KEY` - Groq (backup)

## Milestones

- [x] Physics environment (cannon-es)
- [x] Gap crossing task
- [x] Throw block task
- [x] REST API server
- [x] Python environment wrapper
- [x] LLM policy (Gemini)
- [x] Logging pipeline
- [x] Experiment runner
- [x] Analysis scripts

## Anti-Bloat Rules (v1)

- Only 2 tasks: Gap Crossing + Throw Block
- Only 2 gravity levels: 1g, 0.5g
- Only 3 agent types: RL-F, RL-N, NRL-F
- Single LLM family: Gemini first
- No complex curriculum
- No extra prompt variants
