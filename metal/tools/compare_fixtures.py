#!/usr/bin/env python3
"""Compare strata-metal-run logit dumps with the CUDA fixtures.

Single dump (pick rule on the positions present):

    compare_fixtures.py DUMP PROMPT [--fixtures DIR]

Full gate, every prompt and generation row of p1-p5:

    compare_fixtures.py --gate --mac p1=MAC.bin --pert p1=PERT.bin ...

Per position, each side is normalized by its own full-vocab float64 log-sum-exp.

logprob delta, Mac against CUDA, at CUDA's chosen token t:

    (mac[t] - mac_lse) - (cuda[t] - cuda_lse)

TV distance is half the L1 between those probabilities, summed over CUDA's top
64 ids. Perturbed against Mac uses the same token and the same CUDA top 64:

    (pert[t] - pert_lse) - (mac[t] - mac_lse)

The raw max and mean |mac - cuda| over that top 64 are printed and are not gates.

Picks. The Mac argmax equals CUDA's next_token, or the Mac pick is in CUDA's
top 3 and CUDA's logit gap from its #1 down to that pick is <= 0.5. Any other
mismatch fails. Accepted mismatches are listed with the gap.

p5 prompt NLL. On each of the 306 prompt rows whose actual next prompt token
is in CUDA's top 64, NLL is -(logit - lse) on both sides. The Mac mean must be
within 1% of CUDA's and not more than 1% worse.

Spread. Over every p1-p5 row, the 95th percentile of |logprob delta| and of TV
for Mac against CUDA must each be <= twice the same percentile for perturbed
against Mac. The percentile is Hyndman-Fan type 7 (index 0.95 * (n - 1), linear
between the surrounding ranks).

These cuts are fixed. Do not edit them so a run will pass.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

PICK_GAP = 0.5
PICK_RANK = 3
NLL_BAND = 0.01
SPREAD_FACTOR = 2.0
PROMPTS = ("p1", "p2", "p3", "p4", "p5")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def read_dump(path):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise SystemExit(f"{path}: shorter than a header")
    n_vocab, n_rows = struct.unpack_from("<ii", data, 0)
    if n_vocab <= 0 or n_rows < 0:
        raise SystemExit(f"{path}: bad header vocab={n_vocab} rows={n_rows}")
    need = 8 + n_rows * n_vocab * 4
    if len(data) != need:
        raise SystemExit(f"{path}: {len(data)} bytes, header describes {need}")
    rows = []
    for i in range(n_rows):
        off = 8 + i * n_vocab * 4
        rows.append(struct.unpack_from("<" + str(n_vocab) + "f", data, off))
    return n_vocab, rows


def load_jsonl(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pos = int(row["pos"])
            if pos in rows:
                raise SystemExit(f"{path}: position {pos} is repeated")
            rows[pos] = row
    return rows


def argmax(row):
    best = 0
    top = row[0]
    for i in range(1, len(row)):
        if row[i] > top:
            top = row[i]
            best = i
    return best


def logsumexp(row):
    peak = max(row)
    total = math.fsum(math.exp(value - peak) for value in row)
    return peak + math.log(total)


def percentile_95(values):
    """Hyndman-Fan type 7. index = 0.95 * (n - 1), linear between the ranks."""
    if not values:
        raise SystemExit("a percentile needs at least one value")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = 0.95 * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    weight = rank - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def cuda_top(ref):
    top = ref["top"]
    if len(top) < 3:
        raise SystemExit(f"position {ref.get('pos')}: the fixture top list has {len(top)} entries")
    ids = [int(pair[0]) for pair in top]
    logits = [f32(pair[1]) for pair in top]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"position {ref.get('pos')}: the fixture top list repeats an id")
    order = sorted(range(len(ids)), key=lambda i: (-logits[i], ids[i]))
    ranked_ids = [ids[i] for i in order]
    ranked_logits = [logits[i] for i in order]
    return ranked_ids, ranked_logits


def total_variation(left, left_lse, right_logit, right_lse, ids):
    terms = [
        abs(math.exp(left[token] - left_lse) - math.exp(right_logit[token] - right_lse)) for token in ids
    ]
    return 0.5 * math.fsum(terms)


def load_prompt(name, fixtures):
    meta_path = fixtures / f"{name}.meta.json"
    prompt_path = fixtures / f"{name}.prompt.jsonl"
    gen_path = fixtures / f"{name}.jsonl"
    if not meta_path.is_file() or not prompt_path.is_file() or not gen_path.is_file():
        raise SystemExit(f"missing fixture files for {name} under {fixtures}")
    meta = json.loads(meta_path.read_text())
    tokens = [int(tok) for tok in meta["prompt_tokens"]]
    prompt_rows = load_jsonl(prompt_path)
    gen_rows = load_jsonl(gen_path)
    overlap = set(prompt_rows) & set(gen_rows)
    if overlap:
        raise SystemExit(f"{name}: position {min(overlap)} is in both fixture files")
    expect_prompt = list(range(len(tokens) - 1))
    expect_gen = list(range(len(tokens) - 1, len(tokens) - 1 + 32))
    if sorted(prompt_rows) != expect_prompt:
        raise SystemExit(f"{name}: prompt rows are not positions 0 .. {len(tokens) - 2}")
    if sorted(gen_rows) != expect_gen:
        raise SystemExit(f"{name}: generation rows are not the 32 positions from {expect_gen[0]}")
    for pos, row in prompt_rows.items():
        if int(row["input_token"]) != tokens[pos]:
            raise SystemExit(f"{name} pos {pos}: input token does not match the prompt")
    if int(gen_rows[len(tokens) - 1]["input_token"]) != tokens[-1]:
        raise SystemExit(f"{name}: the first generation row is not the last prompt token")
    return tokens, prompt_rows, gen_rows


def score_position(name, pos, kind, mac, ref, pert, prompt_tokens):
    ranked_ids, ranked_logits = cuda_top(ref)
    cuda_pick = int(ref["next_token"])
    if ranked_ids[0] != cuda_pick:
        raise SystemExit(
            f"{name} pos {pos}: fixture next_token {cuda_pick} is not top-64 #1 {ranked_ids[0]}"
        )
    n_vocab = len(mac)
    if any(token < 0 or token >= n_vocab for token in ranked_ids):
        raise SystemExit(f"{name} pos {pos}: a fixture id is outside vocab {n_vocab}")
    by_id = {token: logit for token, logit in zip(ranked_ids, ranked_logits)}
    if cuda_pick not in by_id:
        raise SystemExit(f"{name} pos {pos}: CUDA's token {cuda_pick} is not in the top 64")
    mac_pick = argmax(mac)
    mac_lse = logsumexp(mac)
    cuda_lse = float(ref["lse"])
    rank = ranked_ids.index(mac_pick) + 1 if mac_pick in by_id else None
    gap = (ranked_logits[0] - by_id[mac_pick]) if mac_pick in by_id else None
    exact = mac_pick == cuda_pick
    accepted = (not exact) and rank is not None and rank <= PICK_RANK and gap <= PICK_GAP
    failed = not exact and not accepted
    diffs = [abs(mac[token] - by_id[token]) for token in ranked_ids]
    logprob = (mac[cuda_pick] - mac_lse) - (by_id[cuda_pick] - cuda_lse)
    tv = total_variation(mac, mac_lse, by_id, cuda_lse, ranked_ids)
    pert_logprob = None
    pert_tv = None
    if pert is not None:
        if len(pert) != n_vocab:
            raise SystemExit(f"{name} pos {pos}: perturbed vocab {len(pert)} != {n_vocab}")
        pert_lse = logsumexp(pert)
        pert_logprob = (pert[cuda_pick] - pert_lse) - (mac[cuda_pick] - mac_lse)
        pert_tv = total_variation(pert, pert_lse, mac, mac_lse, ranked_ids)
    nll_cuda = None
    nll_mac = None
    if kind == "prompt":
        actual = prompt_tokens[pos + 1]
        if actual in by_id:
            nll_cuda = -(by_id[actual] - cuda_lse)
            nll_mac = -(mac[actual] - mac_lse)
    if exact:
        note = "match"
    elif accepted:
        note = "accepted"
    else:
        note = "FAIL pick"
    return {
        "name": name,
        "pos": pos,
        "kind": kind,
        "input": int(ref["input_token"]),
        "cuda": cuda_pick,
        "mac": mac_pick,
        "rank": rank,
        "gap": gap,
        "logprob": logprob,
        "tv": tv,
        "max": max(diffs),
        "mean": sum(diffs) / len(diffs),
        "lse": abs(mac_lse - cuda_lse),
        "pert_logprob": pert_logprob,
        "pert_tv": pert_tv,
        "exact": exact,
        "accepted": accepted,
        "failed": failed,
        "nll_cuda": nll_cuda,
        "nll_mac": nll_mac,
        "note": note,
    }


def score_dump(name, mac_rows, fixtures_rows, tokens, pert_rows):
    scored = []
    for pos in sorted(fixtures_rows):
        if pos < 0 or pos >= len(mac_rows):
            raise SystemExit(f"{name} pos {pos}: the Mac dump has {len(mac_rows)} rows")
        if pert_rows is not None and pos >= len(pert_rows):
            raise SystemExit(f"{name} pos {pos}: the perturbed dump has {len(pert_rows)} rows")
        kind = "prompt" if pos < len(tokens) - 1 else "gen"
        pert = pert_rows[pos] if pert_rows is not None else None
        scored.append(score_position(name, pos, kind, mac_rows[pos], fixtures_rows[pos], pert, tokens))
    return scored


def fmt_num(value, width, digits):
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:{width}.{digits}f}"


def print_rows(title, rows):
    print(f"\n=== {title} ===")
    print(
        f"{'pos':>5} {'input':>8} {'cuda':>8} {'mac':>8} {'rank':>5} {'gap':>10} "
        f"{'logprob':>10} {'tv':>10} {'max|d|':>10} {'mean|d|':>10} {'|lse|':>10}  note"
    )
    for row in rows:
        rank = f"{row['rank']:5d}" if row["rank"] is not None else f"{'-':>5}"
        print(
            f"{row['pos']:5d} {row['input']:8d} {row['cuda']:8d} {row['mac']:8d} {rank} "
            f"{fmt_num(row['gap'], 10, 4)} {row['logprob']:10.4f} {row['tv']:10.4f} "
            f"{row['max']:10.4f} {row['mean']:10.4f} {row['lse']:10.4f}  {row['note']}"
        )
    if not rows:
        print("no positions")
        return
    fails = [row for row in rows if row["failed"]]
    accepted = [row for row in rows if row["accepted"]]
    max_logit = max(row["max"] for row in rows)
    mean_logit = sum(row["mean"] for row in rows) / len(rows)
    max_lse = max(row["lse"] for row in rows)
    mean_lse = sum(row["lse"] for row in rows) / len(rows)
    print(
        f"summary: {len(rows)} positions, {len(rows) - len(fails) - len(accepted)} exact, "
        f"{len(accepted)} accepted mismatches, {len(fails)} pick failures"
    )
    print(
        f"info top-64 |d| max {max_logit:.6f} mean {mean_logit:.6f}  "
        f"|lse| max {max_lse:.6f} mean {mean_lse:.6f}"
    )


def list_accepted(rows):
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        print("accepted mismatches: none")
        return
    print(f"accepted mismatches: {len(accepted)}")
    for row in accepted:
        print(
            f"  {row['name']} pos {row['pos']} cuda {row['cuda']} mac {row['mac']} "
            f"rank {row['rank']} gap {row['gap']:.6f}"
        )


def list_failures(rows):
    failed = [row for row in rows if row["failed"]]
    if not failed:
        print("pick failures: none")
        return failed
    print(f"pick failures: {len(failed)}")
    for row in failed:
        where = "absent" if row["rank"] is None else str(row["rank"])
        gap = "absent" if row["gap"] is None else f"{row['gap']:.6f}"
        print(
            f"  {row['name']} pos {row['pos']} cuda {row['cuda']} mac {row['mac']} "
            f"rank {where} gap {gap}"
        )
    return failed


def nll_report(rows, enforce):
    used = [row for row in rows if row["nll_cuda"] is not None]
    print(f"\n=== p5 prompt NLL ({len(used)} rows whose next prompt token is in CUDA's top 64) ===")
    if not used:
        print("NLL: no rows")
        return False
    cuda_mean = sum(row["nll_cuda"] for row in used) / len(used)
    mac_mean = sum(row["nll_mac"] for row in used) / len(used)
    mac_wins = sum(row["nll_mac"] < row["nll_cuda"] for row in used)
    cuda_wins = sum(row["nll_cuda"] < row["nll_mac"] for row in used)
    ties = len(used) - mac_wins - cuda_wins
    rel = abs(mac_mean - cuda_mean) / cuda_mean
    within = rel <= NLL_BAND
    not_worse = mac_mean <= cuda_mean * (1.0 + NLL_BAND)
    print(f"rows used: {len(used)}")
    print(f"CUDA mean NLL {cuda_mean:.6f}")
    print(f"Mac  mean NLL {mac_mean:.6f}")
    print(f"relative |Mac - CUDA| / CUDA = {rel:.6f} (limit {NLL_BAND:.2f})")
    print(f"Mac wins {mac_wins}, CUDA wins {cuda_wins}, ties {ties}")
    if enforce:
        print(f"NLL within 1%: {'PASS' if within else 'FAIL'}")
        print(f"NLL not more than 1% worse: {'PASS' if not_worse else 'FAIL'}")
    return within and not_worse


def spread_report(rows):
    print("\n=== spread over every scored row ===")
    print("percentile: Hyndman-Fan type 7, index 0.95 * (n - 1)")
    missing = [row for row in rows if row["pert_logprob"] is None or row["pert_tv"] is None]
    if missing:
        raise SystemExit(f"spread is missing a perturbed metric at {missing[0]['name']} pos {missing[0]['pos']}")
    mac_lp = percentile_95([abs(row["logprob"]) for row in rows])
    pert_lp = percentile_95([abs(row["pert_logprob"]) for row in rows])
    mac_tv = percentile_95([row["tv"] for row in rows])
    pert_tv = percentile_95([row["pert_tv"] for row in rows])
    lp_ok = mac_lp <= SPREAD_FACTOR * pert_lp
    tv_ok = mac_tv <= SPREAD_FACTOR * pert_tv
    print(f"rows: {len(rows)}")
    print(
        f"|logprob delta| p95  Mac-CUDA {mac_lp:.6f}  perturbed-Mac {pert_lp:.6f}  "
        f"ratio {mac_lp / pert_lp if pert_lp else float('inf'):.4f}  limit {SPREAD_FACTOR:.1f}  "
        f"{'PASS' if lp_ok else 'FAIL'}"
    )
    print(
        f"TV distance     p95  Mac-CUDA {mac_tv:.6f}  perturbed-Mac {pert_tv:.6f}  "
        f"ratio {mac_tv / pert_tv if pert_tv else float('inf'):.4f}  limit {SPREAD_FACTOR:.1f}  "
        f"{'PASS' if tv_ok else 'FAIL'}"
    )
    print(
        f"noise |logprob delta| median {percentile_median([abs(row['pert_logprob']) for row in rows]):.6f} "
        f"max {max(abs(row['pert_logprob']) for row in rows):.6f}"
    )
    print(
        f"noise TV             median {percentile_median([row['pert_tv'] for row in rows]):.6f} "
        f"max {max(row['pert_tv'] for row in rows):.6f}"
    )
    print(
        f"Mac-CUDA |logprob delta| median {percentile_median([abs(row['logprob']) for row in rows]):.6f} "
        f"max {max(abs(row['logprob']) for row in rows):.6f}"
    )
    print(
        f"Mac-CUDA TV             median {percentile_median([row['tv'] for row in rows]):.6f} "
        f"max {max(row['tv'] for row in rows):.6f}"
    )
    return lp_ok and tv_ok, mac_lp, pert_lp, mac_tv, pert_tv


def percentile_median(values):
    xs = sorted(values)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def parse_named(items, flag):
    found = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{flag} needs NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        if name not in PROMPTS:
            raise SystemExit(f"{flag} name {name} is not one of {', '.join(PROMPTS)}")
        if name in found:
            raise SystemExit(f"{flag} repeats {name}")
        found[name] = Path(path)
    missing = [name for name in PROMPTS if name not in found]
    if missing:
        raise SystemExit(f"{flag} is missing {', '.join(missing)}")
    return found


def run_gate(fixtures, mac_paths, pert_paths):
    print(
        "columns: logprob is (Mac - Mac lse) - (CUDA - CUDA lse) at CUDA's token; "
        "tv sums CUDA's top 64; max|d| and mean|d| are raw logits and are not gates"
    )
    all_rows = []
    by_name = {}
    for name in PROMPTS:
        tokens, prompt_rows, gen_rows = load_prompt(name, fixtures)
        n_vocab, mac_rows = read_dump(mac_paths[name])
        pert_vocab, pert_rows = read_dump(pert_paths[name])
        if pert_vocab != n_vocab:
            raise SystemExit(f"{name}: Mac vocab {n_vocab} != perturbed vocab {pert_vocab}")
        fixtures_rows = {}
        fixtures_rows.update(prompt_rows)
        fixtures_rows.update(gen_rows)
        print(
            f"\n{name}: Mac rows {len(mac_rows)}, perturbed rows {len(pert_rows)}, "
            f"fixture rows {len(fixtures_rows)}, vocab {n_vocab}"
        )
        scored = score_dump(name, mac_rows, fixtures_rows, tokens, pert_rows)
        by_name[name] = scored
        all_rows.extend(scored)

    p1 = [row for row in by_name["p1"] if row["pos"] == 0]
    print_rows("step 1: p1 position 0", p1)
    prompt_small = [row for name in ("p1", "p2", "p3", "p4") for row in by_name[name] if row["kind"] == "prompt"]
    print_rows("step 2: p1-p4 prompt rows", prompt_small)
    p5_prompt = [row for row in by_name["p5"] if row["kind"] == "prompt"]
    print_rows("step 3: p5 prompt rows", p5_prompt)
    gen_rows = [row for row in all_rows if row["kind"] == "gen"]
    print_rows("step 4: generation rows p1-p5", gen_rows)

    print("\n=== criterion 1: picks, every p1-p5 row ===")
    list_accepted(all_rows)
    failed = list_failures(all_rows)
    picks_ok = not failed
    print(f"picks: {'PASS' if picks_ok else 'FAIL'}")

    if len(p5_prompt) != 306:
        raise SystemExit(f"p5 prompt rows scored {len(p5_prompt)}, expected 306")
    nll_ok = nll_report(p5_prompt, enforce=True)
    spread_ok, _, _, _, _ = spread_report(all_rows)
    print("\n=== gate ===")
    print(f"criterion 1 picks: {'PASS' if picks_ok else 'FAIL'}")
    print(f"criterion 2 p5 prompt NLL: {'PASS' if nll_ok else 'FAIL'}")
    print(f"criterion 3 spread: {'PASS' if spread_ok else 'FAIL'}")
    overall = picks_ok and nll_ok and spread_ok
    print(f"result: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


def run_one(dump, prompt, fixtures):
    tokens, prompt_rows, gen_rows = load_prompt(prompt, fixtures)
    n_vocab, mac_rows = read_dump(dump)
    fixtures_rows = {}
    fixtures_rows.update(prompt_rows)
    fixtures_rows.update(gen_rows)
    present = {pos: row for pos, row in fixtures_rows.items() if 0 <= pos < len(mac_rows)}
    missing = sorted(pos for pos in fixtures_rows if pos not in present)
    print(f"dump {dump}: vocab {n_vocab} rows {len(mac_rows)}")
    print(f"{prompt}: {len(present)} positions in both, {len(missing)} fixture positions absent")
    print("spread: not checked, this mode has no perturbed dump")
    if not present:
        print("no overlapping positions")
        return 1
    scored = score_dump(prompt, mac_rows, present, tokens, None)
    print_rows(f"{prompt}", scored)
    print("\n=== picks ===")
    list_accepted(scored)
    failed = list_failures(scored)
    nll_ok = True
    if prompt == "p5":
        prompt_scored = [row for row in scored if row["kind"] == "prompt"]
        if len(prompt_scored) == 306:
            nll_ok = nll_report(prompt_scored, enforce=True)
        else:
            print(f"NLL: {len(prompt_scored)} p5 prompt rows in the dump, the check wants 306")
    print(f"picks: {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed and nll_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Compare logit dumps with the CUDA fixtures.")
    here = Path(__file__).resolve()
    parser.add_argument("dump", nargs="?")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--fixtures", type=Path, default=here.parents[2] / "reference" / "fixtures")
    parser.add_argument("--gate", action="store_true", help="score p1-p5 Mac and perturbed dumps")
    parser.add_argument("--mac", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--pert", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    if args.gate:
        if args.dump or args.prompt:
            raise SystemExit("--gate does not take a positional dump")
        return run_gate(args.fixtures, parse_named(args.mac, "--mac"), parse_named(args.pert, "--pert"))
    if args.mac or args.pert:
        raise SystemExit("--mac and --pert belong to --gate")
    if not args.dump or not args.prompt:
        raise SystemExit("usage: compare_fixtures.py DUMP PROMPT   or   compare_fixtures.py --gate --mac p1=... --pert p1=...")
    if args.prompt not in PROMPTS:
        raise SystemExit(f"prompt {args.prompt} is not one of {', '.join(PROMPTS)}")
    return run_one(args.dump, args.prompt, args.fixtures)


if __name__ == "__main__":
    sys.exit(main())
