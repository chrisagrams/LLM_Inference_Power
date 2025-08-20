import argparse
import random
from pathlib import Path

def parse_indices(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    idx = []
    for p in parts:
        if ":" in p:
            a, b = p.split(":", 1)
            a = int(a) if a else 0
            b = int(b) if b else None
            idx.extend(range(a, b))
        else:
            idx.append(int(p))
    return idx

def main():
    p = argparse.ArgumentParser(description="Make matching subsets of two line-aligned files.")
    p.add_argument("inputs_file")
    p.add_argument("lengths_file")
    p.add_argument("n", type=int)
    p.add_argument("out_inputs_file")
    p.add_argument("out_lengths_file")
    p.add_argument("--indices", help="Comma list of 0-based indices; supports slices like 10:20.")
    p.add_argument("--indices-file", help="File with one 0-based index per line.")
    p.add_argument("--random", action="store_true", help="Sample n indices uniformly without replacement.")
    p.add_argument("--seed", type=int, help="Seed for --random.")
    args = p.parse_args()

    in1 = Path(args.inputs_file).read_text(encoding="utf-8").splitlines(keepends=True)
    in2 = Path(args.lengths_file).read_text(encoding="utf-8").splitlines(keepends=True)

    if len(in1) != len(in2):
        raise SystemExit(f"Input files have different lengths: {len(in1)} vs {len(in2)}")
    total = len(in1)
    if not (0 < args.n <= total):
        raise SystemExit(f"n must be in 1..{total}, got {args.n}")

    if args.indices and (args.random or args.indices_file):
        raise SystemExit("Provide only one of --indices, --indices-file, or --random.")
    if args.indices_file and args.random:
        raise SystemExit("Provide only one of --indices, --indices-file, or --random.")

    if args.indices:
        sel = parse_indices(args.indices)
    elif args.indices_file:
        raw = Path(args.indices_file).read_text(encoding="utf-8").splitlines()
        sel = [int(x.strip()) for x in raw if x.strip()]
    elif args.random:
        rng = random.Random(args.seed)
        sel = rng.sample(range(total), args.n)
    else:
        sel = list(range(args.n))

    if len(sel) != args.n:
        raise SystemExit(f"Selected {len(sel)} indices, expected n={args.n}")
    if any(i < 0 or i >= total for i in sel):
        bad = [i for i in sel if i < 0 or i >= total][:5]
        raise SystemExit(f"Out-of-range indices (showing up to 5): {bad}")
    if len(set(sel)) != len(sel):
        raise SystemExit("Duplicate indices are not allowed.")

    out1 = [in1[i] for i in sel]
    out2 = [in2[i] for i in sel]

    Path(args.out_inputs_file).write_text("".join(out1), encoding="utf-8")
    Path(args.out_lengths_file).write_text("".join(out2), encoding="utf-8")

if __name__ == "__main__":
    main()
