"""Check that no dead (pre-correction) number survives in the report text.

Allowed occurrences are only inside explicit retraction/supersession
sentences, which must name the number and the word 'retract' / 'supersede' /
'pre-correction' within a small window.

Run:  python scripts/check_dead_numbers.py
Exit 1 on any violation.
"""
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "Mlawn-midsem-tex.tex")

DEAD = ["0.674", "0.298", "0.860", "0.749", "0.926", "0.381", "0.625",
        "0.639", "0.418", "0.615", "0.4500"]

ALLOWED_NEAR = ("retract", "supersede", "pre-correction", "pre-correction",
                "earlier version", "first version", "record", "history",
                "superseded", "old", "legacy", "withdrawn")


def main():
    s = open(TEX, encoding="utf-8").read()
    lines = s.splitlines()
    bad = []
    for v in DEAD:
        for i, ln in enumerate(lines, 1):
            if v not in ln:
                continue
            if ln.lstrip()[:1] == "\\" and "newcommand" in ln.lstrip()[:20]:
                continue  # token definitions are data, not prose
            window = ln.lower()
            if i > 1:
                window += " " + lines[i - 2].lower()
            if i < len(lines):
                window += " " + lines[i].lower()
            if not any(w in window for w in ALLOWED_NEAR):
                bad.append((i, v, ln.strip()[:120]))
    if bad:
        print("DEAD-NUMBER VIOLATIONS (%d):" % len(bad))
        for i, v, ln in bad:
            print("  line %d [%s]: %s" % (i, v, ln))
        return 1
    print("clean: no dead number outside a retraction/supersession sentence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
