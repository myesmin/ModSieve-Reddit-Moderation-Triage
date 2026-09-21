"""Feasibility probe: are moderator removals observable through the public API?

    python scripts/probe_removals.py snapshot     # record the newest posts now
    python scripts/probe_removals.py recheck      # hours later, look them up by id

Reddit hides removed posts from listings, so a removal can only be observed by
recording a post while it is live and looking it up again later. Only posts
that were young when snapshotted count: a post first seen at five days old has
already survived five days of moderation, and would bias the removal rate
towards zero.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import praw
from dotenv import load_dotenv

from src.collect import DEFAULT_SUBREDDITS

SNAPSHOT = ROOT / "Data" / "probe" / "snapshot.json"
RECHECK = ROOT / "Data" / "probe" / "recheck.json"
YOUNG_HOURS = 3


def reddit():
    load_dotenv(ROOT / ".env")
    return praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                       client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                       user_agent=os.environ.get("REDDIT_USER_AGENT", "triage-probe"))


def snapshot():
    r, now, rows = reddit(), time.time(), []
    for sub in DEFAULT_SUBREDDITS:
        for p in r.subreddit(sub).new(limit=100):
            rows.append({"id": p.id, "subreddit": sub, "created_utc": p.created_utc,
                         "snapshot_utc": now, "age_hours": (now - p.created_utc) / 3600,
                         "title": p.title, "has_body": bool((p.selftext or "").strip())})
    SNAPSHOT.write_text(json.dumps(rows))
    young = [x for x in rows if x["age_hours"] < YOUNG_HOURS]
    print(f"snapshotted {len(rows)} posts; {len(young)} were under {YOUNG_HOURS}h old")
    print("young per community:", dict(Counter(x["subreddit"] for x in young)))


def recheck():
    rows = {x["id"]: x for x in json.loads(SNAPSHOT.read_text())}
    r, now, out = reddit(), time.time(), []
    ids = list(rows)
    for i in range(0, len(ids), 100):
        for p in r.info(fullnames=[f"t3_{x}" for x in ids[i:i + 100]]):
            base = rows[p.id]
            out.append({**base, "checked_utc": now,
                        "hours_observed": (now - base["snapshot_utc"]) / 3600,
                        "removed_by_category": getattr(p, "removed_by_category", None),
                        "selftext_marker": p.selftext if p.selftext in ("[removed]", "[deleted]") else "",
                        "author_gone": p.author is None,
                        "robot_indexable": getattr(p, "is_robot_indexable", None)})
    RECHECK.write_text(json.dumps(out))

    young = [x for x in out if x["age_hours"] < YOUNG_HOURS]
    print(f"rechecked {len(out)} of {len(rows)} posts after "
          f"{out[0]['hours_observed']:.1f}h" if out else "nothing returned")
    print("removed_by_category, all posts:", dict(Counter(x["removed_by_category"] for x in out)))
    print(f"removed_by_category, posts <{YOUNG_HOURS}h old at snapshot (n={len(young)}):",
          dict(Counter(x["removed_by_category"] for x in young)))
    print("selftext markers:", dict(Counter(x["selftext_marker"] for x in out)))
    print("not robot-indexable:", sum(x["robot_indexable"] is False for x in out))
    for x in [x for x in out if x["removed_by_category"]][:8]:
        print(f"  [{x['subreddit']}] {x['removed_by_category']:<18} {x['title'][:70]}")


if __name__ == "__main__":
    {"snapshot": snapshot, "recheck": recheck}[sys.argv[1]]()
