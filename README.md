---
title: Restaurant Table Management
emoji: 🍽️
colorForm: blue
colorTO: green
sdk: docker
pinned: false
---
# 🍽️ Restaurant Table Management – OpenEnv RL Environment

> **OpenEnv Hackathon × SCALER** | Production-ready Reinforcement Learning Environment

A fully compliant OpenEnv environment where an agent learns to manage restaurant seating in real-time – balancing customer satisfaction, table utilisation, and revenue across three difficulty levels.

---

## 🧠 Environment Design

### The RL Loop
```
observe → act → reward → repeat
```

### Observation Space
| Field | Type | Description |
|---|---|---|
| `tables` | List[Table] | Each table: id, capacity, status, party_size, time_seated |
| `waiting_queue` | List[Customer] | Each customer: party_size, patience_remaining, revenue_value |
| `time_step` | int | Current episode step |
| `occupancy_rate` | float | Fraction of tables occupied [0,1] |
| `total_seated` | int | Customers successfully seated |
| `total_rejected` | int | Customers manually rejected |
| `total_walkouts` | int | Customers who left due to impatience |
| `total_revenue` | float | Episode revenue so far |

### Action Space (4 discrete actions)
| Action | Description | Reward Impact |
|---|---|---|
| `assign_table` | Seat next customer at best-fit table | +10 to +18 (efficiency bonus) |
| `reject_customer` | Remove first customer from queue | −5 (penalty) |
| `delay_seating` | Wait one step | −0.5 (small penalty) |
| `combine_tables` | Merge two small tables for large party | +8 to +14 |

### Reward Function
```
R = seat_reward + utilisation_bonus − walkout_penalty − idle_penalty − delay_penalty
  + end_bonus (satisfaction rate × 10)
```
- **Positive**: efficient seating, high occupancy, speed bonus, revenue
- **Negative**: walkouts (−8 each), rejections (−5 each), idle tables, delays

---

## 📋 Tasks

| Task | Tables | Arrival Rate | Patience | Steps | Description |
|---|---|---|---|---|---|
| `easy` | 6 | 30% | 8 steps | 100 | Low traffic, simple decisions |
| `medium` | 10 | 55% | 5 steps | 150 | Moderate traffic, mixed sizes |
| `hard` | 14 | 80% | 3 steps | 200 | Peak hours, high randomness |

---

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone and install
git clone <repo-url>
cd my_env_v4-main
pip install -r requirements.txt

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Run inference (in another terminal)
export API_BASE_URL=http://localhost:8000
export MODEL_NAME=claude-sonnet-4-20250514
export HF_TOKEN=your_token_here
python inference.py --task easy --seed 42

# 4. Grade all tasks
python grader.py --task all --runs 1
```

### Docker
```bash
# Build
docker build -t restaurant-env .

# Run
docker run -p 8000:8000 \
  -e MODEL_NAME=claude-sonnet-4-20250514 \
  -e HF_TOKEN=your_token \
  restaurant-env

# Test health
curl http://localhost:8000/health
```

---

## 🌐 API Reference

### `POST /reset`
Reset the environment to a clean state.
```json
{
  "task": "easy",
  "seed": 42
}
```

### `POST /step`
Advance one step with an action.
```json
{
  "action": "assign_table",
  "table_id": null,
  "combine_with": null
}
```

### `GET /state`
Get current state without advancing step.

### `GET /health`
Returns `{"status": "healthy", "version": "1.0.0"}` with HTTP 200.

---

## 📊 Grader

The grader evaluates three dimensions with a weighted harmonic mean:

| Dimension | Weight | Measures |
|---|---|---|
| Efficiency | 35% | Table occupancy rate |
| Revenue | 35% | Revenue vs theoretical max |
| Satisfaction | 30% | Seated / (seated + rejected + walkouts) |

Scores are always in **(0.0, 1.0)** – never constant, never exactly 0 or 1.

```bash
# Grade a single task
python grader.py --task easy --seed 42

# Grade all tasks
python grader.py --task all --runs 3
```

---

## 📁 File Structure

```
my_env_v4-main/
├── openenv.yaml              # OpenEnv spec (tasks, spaces, rewards)
├── models.py                 # Pydantic typed models
├── inference.py              # LLM-powered baseline agent
├── grader.py                 # Deterministic grader
├── client.py                 # CLI HTTP client
├── requirements.txt
├── Dockerfile
├── .env                      # (git-ignored) environment variables
├── README.md
└── server/
    ├── app.py                # FastAPI REST API
    ├── environment.py        # Core RL environment
    └── my_env_v4_environment.py
```

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | Running server URL |
| `MODEL_NAME` | Yes | LLM model name |
| `HF_TOKEN` | Yes (HF Spaces) | Hugging Face token |
| `OPENAI_API_KEY` | Yes | API key for LLM calls |
| `OPENAI_BASE_URL` | Optional | Override LLM endpoint |

---

## 📝 Log Format

All logs strictly follow:
```
[START] task=easy seed=42 max_steps=100 tables=6
[STEP] t=1 action=assign_table reward=12.50 occ=0.17 queue=2 done=False
[END] steps=100 seated=28 rejected=2 walkouts=3 revenue=1820.50
```

---

## ⏱️ Constraints

- Runtime: < 20 minutes per full evaluation
- Compute: 2 vCPU, 8 GB RAM (Hugging Face Spaces compatible)
- Python: 3.11+

---

## 🏆 Scoring Rubric

| Criterion | Points |
|---|---|
| Environment deploys and API returns 200 | Required |
| reset() works correctly | Required |
| Reward function non-constant and meaningful | 25 pts |
| Grader scores variable (not constant) | 25 pts |
| inference.py runs without errors | 20 pts |
| 3 tasks with increasing difficulty | 15 pts |
| Code quality, comments, README | 15 pts |
