from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'cache'
RESULTS_DIR = ROOT / 'results'
REPORT_DIR = ROOT / 'report'

GRID_SIZE = 9
PATH_LENGTH = 17
DISTANCE_CAP = 3.0

# true reward = GOAL_WEIGHT * distance to goal + HAZARD_WEIGHT * capped distance to the hazard
GOAL_WEIGHT = -1.0
HAZARD_WEIGHT = 8.0

N_DEMOS = 10
DEMO_SEED_OFFSET = 1000
NEW_LAYOUT_SEED_OFFSET = 100

# FLEX-V asks the user when the two BIC scores are closer than this
DELTA = 2.0

OLLAMA_HOST = 'http://localhost:11434'
LLM_TIMEOUT = 600
