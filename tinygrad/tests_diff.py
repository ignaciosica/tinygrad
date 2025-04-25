#!/usr/bin/env python3
import json, subprocess, sys, pathlib
from tabulate import tabulate

def collect(repo):
  """Run pytest, get a map test-node-id → status (normal|skip|xfail)."""
  subprocess.run(
    [
      sys.executable,
      "-m",
      "pytest",
      "-q",  # quiet run
      "--maxfail=0",
      "--color=yes",
      "--json-report",
      "--json-report-file=report.json",
      "-m",
      "not benchmark",
    ],  # skip heavy markers if you have them
    cwd=repo,
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )
  data = json.loads(pathlib.Path(repo, "report.json").read_text())
  status = {}
  for entry in data["tests"]:
    nid = entry["nodeid"]
    outcome = entry["outcome"]  # passed, skipped, xfailed, etc.
    if outcome.startswith("x"):
      status[nid] = "xfail"
    elif outcome == "skipped":
      status[nid] = "skip"
    else:
      status[nid] = "normal"
  return status


if len(sys.argv) < 3:
  here = pathlib.Path(__file__).resolve().parent
  base = here / "base"
  pr   = here / "pr"
else:
  base, pr = sys.argv[1:]
base_map, pr_map = collect(base), collect(pr)

added = [k for k in pr_map if k not in base_map]
removed = [k for k in base_map if k not in pr_map]

changed = []
for test in pr_map:
  if test in base_map and pr_map[test] != base_map[test]:
    changed.append((test, base_map[test], pr_map[test]))


def fmt(lst, hdr):
  return "\n".join([f"### {hdr} ({len(lst)})"] + [""] + ["```"] + lst + ["```"]) if lst else ""


tbl = tabulate([(t, b, p) for t, b, p in changed], headers=["Test", "master", "PR"], tablefmt="github") if changed else ""

msg = "\n\n".join(
  s
  for s in ["## Pytest suite diff vs master", fmt(added, "New tests"), fmt(removed, "Removed tests"), "### Status changes" if changed else "", tbl]
  if s
)

# GitHub-env var consumed by workflow step
print("TEST_DIFF<<EOF")
print(msg or "No differences detected. ✅")
print("EOF")
