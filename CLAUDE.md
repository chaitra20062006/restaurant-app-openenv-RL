# CLAUDE.md – AI Context for Restaurant Table Management OpenEnv

## Project Summary
Production-ready OpenEnv RL environment for a restaurant table management system.
Implements the observe → act → reward → repeat loop with full OpenEnv spec compliance.

## File Structure
```
my_env_v4-main/
├── openenv.yaml              # OpenEnv spec metadata (tasks, spaces, rewards)
├── models.py                 # Pydantic typed models (Table, Customer, ObservationState, etc.)
├── inference.py              # Baseline LLM-powered agent (uses OpenAI client)
├── grader.py                 # Deterministic grader, scores in (0.0, 1.0)
├── client.py                 # CLI HTTP client for manual testing
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production Docker image
├── .env                      # Environment variables (git-ignored)
├── .gitignore
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                # FastAPI REST API (reset, step, state, health)
    ├── environment.py        # Core RL environment logic
    └── my_env_v4_environment.py  # OpenEnv registration shim
```

## Key Design Decisions
- **4 discrete actions**: assign_table, reject_customer, delay_seating, combine_tables
- **Shaped rewards**: positive for seating efficiency, negative for walkouts/idle
- **3 tasks**: easy (6 tables), medium (10 tables), hard (14 tables, peak hours)
- **Grader**: weighted harmonic mean of efficiency, revenue, satisfaction scores
- **Logs**: strictly [START], [STEP], [END] format

## Environment Variables Required
- `API_BASE_URL` – running server URL
- `MODEL_NAME` – LLM model name
- `HF_TOKEN` – Hugging Face token for deployment

## Running Locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --reload --port 8000
python inference.py --task easy
python grader.py --task all
```
