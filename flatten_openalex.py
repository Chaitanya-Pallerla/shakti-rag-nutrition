"""
flatten_openalex.py: turn the OpenAlex works snapshot layout into the flat folder full_index.py reads.

  aws s3 sync s3://openalex/data/works <download> --no-sign-request
  python flatten_openalex.py <download>          -> <SHAKTI_RESEARCH_DIR>/openalex_works/updated_date=..._part_NNNN.gz

<download>/updated_date=2016-06-24/part_0000.gz becomes openalex_works/updated_date=2016-06-24_part_0000.gz.
Files are moved (same drive: instant, no extra space); pass --copy to copy instead.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import glob
import shutil


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    copy = "--copy" in sys.argv
    out = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "openalex_works")
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(args[0], "updated_date=*", "part_*.gz"))):
        dst = os.path.join(out, f"{os.path.basename(os.path.dirname(f))}_{os.path.basename(f)}")
        if os.path.exists(dst):
            continue
        (shutil.copy2 if copy else shutil.move)(f, dst)
        n += 1
    print(f"{n} files -> {out}")


if __name__ == "__main__":
    main()
