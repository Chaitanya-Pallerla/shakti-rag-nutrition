"""
shakti_config.py: loads your settings (folders, model choice, API keys) from ONE file, shakti.env.

  1. copy shakti.env.example to shakti.env      (shakti.env is in .gitignore: it is never committed)
  2. paste your paths and keys into shakti.env
  3. run any script as usual; every script imports this module first

Lines are KEY=value (quotes optional, # starts a comment). A variable already set in your shell wins over
the file, so you can still override one setting for one run. Another file: set SHAKTI_ENV_FILE=path.
Relative paths in the file are taken relative to the folder that holds the file.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.environ.get("SHAKTI_ENV_FILE", os.path.join(HERE, "shakti.env"))
PATH_KEYS = {"SHAKTI_RESEARCH_DIR", "SHAKTI_SERVE_INDEX", "SHAKTI_FOOD_DB", "SHAKTI_MODELS_DIR", "HF_HOME"}


def load(path=ENV_FILE):
    if not os.path.exists(path):
        return {}
    base, got = os.path.dirname(os.path.abspath(path)), {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.split(" #", 1)[0].strip().strip('"').strip("'")
            if not v:
                continue                                  # empty = keep the built-in default
            if k in PATH_KEYS and not os.path.isabs(v):
                v = os.path.normpath(os.path.join(base, v))
            got[k] = v
            os.environ.setdefault(k, v)
    return got


LOADED = load()

if __name__ == "__main__":                               # python shakti_config.py -> show what is set (keys masked)
    for k, v in sorted(LOADED.items()):
        secret = any(s in k for s in ("KEY", "TOKEN", "SECRET"))
        print(f"{k} = {(v[:4] + '...' + v[-2:]) if secret and len(v) > 8 else ('***' if secret else os.environ[k])}")
    print(f"({len(LOADED)} settings from {ENV_FILE})" if LOADED else f"no {ENV_FILE}: copy shakti.env.example first")
