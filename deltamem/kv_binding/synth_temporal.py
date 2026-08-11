"""Synthetic TEMPORAL episodes in LoCoMo cat2 style: a short timestamped dialogue where a person
mentions doing something, and a "When did X?" question whose answer is the CLEAN date. Teaches the
model to EXTRACT a clean date/relative-time span instead of dumping the raw conversation timestamp
("date: 7:18pm on 27 may 2023") -- the measured cat2 failure mode. memalpha temporal QA does not
fit our 4500-ctx budget (only ~2 examples), so we synthesize controllable, format-matched data.
"""
from __future__ import annotations
import random

_NAMES = ["Alice", "Ben", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
          "Karen", "Leo", "Mia", "Nate", "Olivia", "Paul", "Quinn", "Rose", "Sam", "Tina"]
_EVENTS = ["went to the museum", "started a new job", "adopted a puppy", "moved to a new city",
           "ran a marathon", "took a cooking class", "visited their grandparents", "got a promotion",
           "started learning guitar", "joined a gym", "went camping", "had a birthday party",
           "began a pottery course", "traveled to Rome", "planted a garden", "adopted a cat",
           "started volunteering", "went on a road trip", "attended a concert", "opened a bakery"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]
_WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _fmt_date(rng, y):
    """Return (spoken_in_dialogue, gold_answer) for a date, in a LoCoMo-like variety of granularities."""
    m = rng.randint(1, 12); d = rng.randint(1, 28)
    mn = _MONTHS[m - 1]
    style = rng.randint(0, 4)
    if style == 0:   # full date
        return f"{d} {mn} {y}", f"{d} {mn} {y}"
    if style == 1:   # month + year
        return f"in {mn} {y}", f"{mn} {y}"
    if style == 2:   # year only
        return f"back in {y}", f"{y}"
    if style == 3:   # relative "N weeks ago" (dialogue anchored)
        n = rng.randint(2, 8)
        return f"about {n} weeks ago", f"{n} weeks ago"
    # weekday-relative
    wd = rng.choice(_WD)
    return f"last {wd}", f"{wd}"


def build_synthetic_temporal_episodes(n_episodes=60, queries_per_ep=4, seed=0):
    rng = random.Random(seed)
    eps = []
    for _ in range(n_episodes):
        y = rng.choice([2021, 2022, 2023])
        people = rng.sample(_NAMES, k=queries_per_ep)
        events = rng.sample(_EVENTS, k=queries_per_ep)
        lines, queries = [], []
        for person, event in zip(people, events):
            spoken, gold = _fmt_date(rng, y)
            lines.append(f"[Dialogue on {rng.randint(1,28)} {rng.choice(_MONTHS)} {y}] "
                         f"{person}: I {event} {spoken}.")
            queries.append({"question": f"When did {person} {event.split()[0] and event}?"
                            .replace("  ", " "),
                            "answer": gold})
        # simplify the question phrasing to "When did <Name> <event>?"
        queries = [{"question": f"When did {p} {e}?", "answer": q["answer"]}
                   for (p, e, q) in zip(people, events, queries)]
        rng.shuffle(lines)
        eps.append({"chunks": lines, "queries": queries})
    return eps


if __name__ == "__main__":
    e = build_synthetic_temporal_episodes(3, 4, seed=1)
    for ep in e[:1]:
        print("CONTEXT:"); [print("  " + c) for c in ep["chunks"]]
        print("QA:"); [print("  ", q) for q in ep["queries"]]
