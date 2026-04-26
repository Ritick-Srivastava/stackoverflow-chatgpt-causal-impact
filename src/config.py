from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# ChatGPT public launch — first post-intervention month is Dec 2022
INTERVENTION_DATE = "2022-11-01"
PRE_PERIOD  = ["2018-01-01", "2022-10-01"]
POST_PERIOD = ["2022-11-01", "2026-04-01"]

START_DATE = "2018-01-01"
END_DATE   = "2026-04-01"

TREATMENT_KEYWORD = "stack overflow"
CONTROL_KEYWORDS  = ["github", "w3schools", "geeksforgeeks"]
ALL_KEYWORDS      = [TREATMENT_KEYWORD] + CONTROL_KEYWORDS

# GitHub excluded from causal model — AI/Copilot growth contaminates it post-2022
CAUSAL_CONTROLS   = ["w3schools", "geeksforgeeks"]

# Human-readable column labels
KEYWORD_LABELS = {
    "stack overflow": "StackOverflow",
    "github":         "GitHub",
    "w3schools":      "W3Schools",
    "geeksforgeeks":  "GeeksForGeeks",
}
