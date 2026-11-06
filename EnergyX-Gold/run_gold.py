"""Run every gold task and write outputs/gold_results.jsonl.

    python run_gold.py

The same as `python -m energyx_gold.cli.run_all`.
"""

from energyx_gold.cli.run_all import main

if __name__ == "__main__":
    main()
