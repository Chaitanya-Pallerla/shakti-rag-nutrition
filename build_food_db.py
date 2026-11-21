"""
build_food_db.py: build food.duckdb (the USDA FoodData Central tables the chat server reads) from the
official FDC CSV download.

  1. Download "Full Download of All Data Types" (CSV) from https://fdc.nal.usda.gov/download-datasets
     and unzip it (folder with food.csv, food_nutrient.csv, food_portion.csv, ...).
  2. python build_food_db.py <unzipped folder> [out.duckdb]     (default out: SHAKTI_FOOD_DB or food.duckdb)

Tables: usda_food(fdc_id, data_type, description, ...), usda_food_nutrient(fdc_id, nutrient_id, amount, ...),
usda_food_portion(fdc_id, amount, portion_description, modifier, gram_weight, ...), usda_nutrient.
FoodData Central is public domain (U.S. Department of Agriculture, Agricultural Research Service).
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import time

import duckdb

TABLES = {"usda_food": "food.csv", "usda_food_nutrient": "food_nutrient.csv",
          "usda_food_portion": "food_portion.csv", "usda_nutrient": "nutrient.csv"}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SHAKTI_FOOD_DB", "food.duckdb")
    t = time.time()
    con = duckdb.connect(out)
    for table, csv in TABLES.items():
        p = os.path.join(src, csv)
        if not os.path.exists(p):
            sys.exit(f"missing {p}: point to the unzipped FoodData Central CSV folder")
        con.execute(f"create or replace table {table} as select * from read_csv_auto(?, header=true, "
                    f"all_varchar=false, sample_size=-1)", [p])
        print(f"{table}: {con.execute(f'select count(*) from {table}').fetchone()[0]:,} rows")
    con.execute("create index if not exists i_fn on usda_food_nutrient(fdc_id)")
    con.execute("create index if not exists i_fp on usda_food_portion(fdc_id)")
    con.close()
    print(f"-> {out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
