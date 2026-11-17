"""Step 1: download GH Archive hour files and keep only the cohort's events.

For each hour, the raw file is downloaded, filtered with the course script's own
`preprocess.extract_hour` (cohort actor, the 6 event types, no bots) and written back under
the same file name. Running scripts/preprocess.py on the filtered folder gives the same tables
as on the raw files, with ~1% of the disk space. Resumable: finished hours are skipped.
"""
import argparse
import gzip
import os
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preprocess as P  # noqa: E402

URL = "https://data.gharchive.org/{}.json.gz"


def _init(cohort):
    P._init_worker(cohort)


def one(args):
    name, out_dir = args
    dst = Path(out_dir) / f"{name}.json.gz"
    miss = Path(out_dir) / f"{name}.missing"
    if dst.exists() or miss.exists():
        return name, "skip"
    tmp = Path(out_dir) / f".{name}.{os.getpid()}.raw"
    for attempt in range(5):
        r = subprocess.run(["curl", "-sS", "-o", str(tmp), "-w", "%{http_code}", URL.format(name)],
                           capture_output=True, text=True)
        if r.stdout == "200":
            break
        if r.stdout == "404":
            miss.write_text("404")
            tmp.unlink(missing_ok=True)
            return name, "404"
        time.sleep(5 * (attempt + 1))
    else:
        return name, "fail " + r.stdout + r.stderr[-100:]
    lines = P.extract_hour(tmp)
    part = dst.with_suffix(".part")
    with gzip.open(part, "wb") as f:
        for line in lines:
            f.write(line + b"\n")
    part.rename(dst)
    tmp.unlink()
    return name, f"ok {len(lines)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-12-01")
    ap.add_argument("--end", required=True, help="UTC hour 'YYYY-MM-DD HH', exclusive")
    ap.add_argument("--out", default="gharchive_cohort")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    h = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    names = []
    while h < end:
        names.append(f"{h:%Y-%m-%d}-{h.hour}")
        h += timedelta(hours=1)
    cohort = P.load_cohort(Path(P.__file__).with_name("cohort.txt"))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cohort,)) as ex:
        for i, (n, s) in enumerate(ex.map(one, [(n, a.out) for n in names]), 1):
            if s != "skip" or i % 500 == 0:
                print(f"{i}/{len(names)} {n} {s}", flush=True)


if __name__ == "__main__":
    main()
