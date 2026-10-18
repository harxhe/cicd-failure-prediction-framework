"""
generate_builds.py: push a controlled series of commits to the SAMPLE APP repo so
every build produces real, varied data, and so the demo can be staged.

Run it from inside the sample app repo (the one CodePipeline watches),
never from the framework repo. It refuses to run anywhere else.

  python generate_builds.py history --dry-run     preview the plan, change nothing
  python generate_builds.py history               push the whole history (about 25 min)
  python generate_builds.py history --start 13    resume at step 13 after an interruption
  python generate_builds.py one --mode STABLE --files 2 --lines 40
                                                  push a single commit (the live demo push)

Modes are stored in fail_flag.txt and read by the tests:
  STABLE  the build passes (the flaky test is switched off)
  FAIL    the build fails on purpose
  FLAKY   the flaky test is on: roughly a 30% chance the build fails
"""
import argparse
import os
import subprocess
import sys
import time
import uuid

REPO_MARKER = "cicd-failure-prediction-sample"
FLAG = "fail_flag.txt"
MODES = ("STABLE", "FAIL", "FLAKY")

LEVER_TEST = '''import os


def _mode():
    flag = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fail_flag.txt")
    if os.path.exists(flag):
        with open(flag) as handle:
            return handle.read().strip()
    return "FLAKY"


def test_forced_outcome():
    """Demo lever: the build is forced to fail when fail_flag.txt says FAIL."""
    assert _mode() != "FAIL"
'''

FLAKY_TEST = '''import os
import random


def _mode():
    flag = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fail_flag.txt")
    if os.path.exists(flag):
        with open(flag) as handle:
            return handle.read().strip()
    return "FLAKY"


def test_flaky_timing():
    # Only flaky when fail_flag.txt says FLAKY (or is absent); otherwise switched off.
    if _mode() != "FLAKY":
        return
    assert random.random() > 0.3
'''

# (mode, number of new files, total lines added across them)
HISTORY = [
    ("STABLE", 1, 10), ("STABLE", 2, 40), ("FLAKY", 1, 15), ("STABLE", 3, 90),
    ("FLAKY", 2, 50), ("FAIL", 2, 60), ("STABLE", 1, 8), ("FAIL", 4, 200),
    ("STABLE", 2, 30), ("FLAKY", 1, 12), ("STABLE", 5, 250), ("STABLE", 1, 5),
    # failure streak: after these five, the next build should score HIGH
    ("FAIL", 2, 50), ("FAIL", 3, 120), ("FAIL", 1, 20), ("FAIL", 4, 180), ("FAIL", 2, 60),
]


def git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def write(path, text):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write(text)


def preflight():
    os.chdir(git("rev-parse", "--show-toplevel"))
    url = git("remote", "get-url", "origin")
    if REPO_MARKER not in url or "framework" in url:
        sys.exit(f"Wrong repo: origin is {url}\nRun this from the sample app repo.")
    if git("branch", "--show-current") != "main":
        sys.exit("Switch to the main branch first.")
    if git("status", "--porcelain"):
        sys.exit("The working tree is not clean. Commit or stash your changes first.")
    git("pull", "--ff-only")


def commit_and_push(message):
    git("add", "-A")
    git("commit", "-m", message)
    git("push")
    print(f"  pushed: {message}", flush=True)


def ensure_lever():
    if os.path.exists("tests/test_lever.py") and os.path.exists(FLAG):
        return
    print("Adding the outcome lever (one setup commit)...", flush=True)
    write("tests/test_lever.py", LEVER_TEST)
    write("tests/test_flaky.py", FLAKY_TEST)
    write(FLAG, "STABLE\n")
    commit_and_push("test: add outcome lever for controlled builds")


def make_build(mode, files, total_lines, label):
    token = uuid.uuid4().hex[:6]
    counts = [total_lines // files + (1 if i < total_lines % files else 0) for i in range(files)]
    for i, count in enumerate(counts):
        body = "".join(f"VALUE_{token}_{i}_{j} = {j}\n" for j in range(count))
        write(f"gen/mod_{token}_{i}.py", body)
    write(FLAG, mode + "\n")
    commit_and_push(f"{label}: {mode.lower()}, {files} files, {total_lines} lines")


def run_history(args):
    steps = HISTORY[args.start - 1:]
    print(f"Plan: {len(steps)} builds, starting at step {args.start}\n")
    for n, (mode, files, lines) in enumerate(HISTORY, start=1):
        marker = "  " if n >= args.start else "x "
        print(f"{marker}{n:>2}. {mode:<6} {files} files, {lines:>3} lines")
    print(f"\nEstimated time: about {len(steps) * (args.wait + 5) // 60} minutes")
    if args.dry_run:
        return
    if not args.yes and input(f"\nPush {len(steps)} commits? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")
    preflight()
    ensure_lever()
    for n, (mode, files, lines) in enumerate(HISTORY, start=1):
        if n < args.start:
            continue
        make_build(mode, files, lines, f"build {n}/{len(HISTORY)}")
        if n < len(HISTORY):
            time.sleep(args.wait)
    print("\nDone. Wait about 90 seconds for the last build before pushing anything else.")


def run_one(args):
    preflight()
    ensure_lever()
    make_build(args.mode, args.files, args.lines, "demo push")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    hist = sub.add_parser("history")
    hist.add_argument("--wait", type=int, default=90, help="seconds between pushes (default 90)")
    hist.add_argument("--start", type=int, default=1, help="resume at this step")
    hist.add_argument("--dry-run", action="store_true")
    hist.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    one = sub.add_parser("one")
    one.add_argument("--mode", choices=MODES, required=True)
    one.add_argument("--files", type=int, default=1)
    one.add_argument("--lines", type=int, default=10, help="total lines across the new files")

    args = parser.parse_args()
    if args.cmd == "history":
        run_history(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
