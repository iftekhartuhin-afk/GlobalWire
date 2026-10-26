#!/usr/bin/env python3
"""RSS -> original AI brief (Gemini free tier) -> data/articles.json.
Uses ONLY the feed's own title + description. Never scrapes full articles."""
import json, os, re, time, html, hashlib, difflib, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG, OUT, STATUS = (os.path.join(ROOT, p) for p in ("config/sources.json", "data/articles.json", "data/status.json"))
CATS = ["World", "Politics", "Business", "Technology", "Science", "Health", "Environment", "Sports", "Culture"]
KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
UA = "NewsAggregatorBot/1.0 (headline+link aggregator)"

def get(url, timeout=25, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(t or ""))).strip()

def iso(s):
    try:
        d = parsedate_to_datetime(s)
    except Exception:
        try: d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception: return datetime.now(timezone.utc).isoformat()
    if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat()

def parse_feed(raw):
    out = []
    for el in ET.fromstring(raw).iter():
        if el.tag.split("}")[-1] not in ("item", "entry"): continue
        def f(*names):
            for c in el:
                k = c.tag.split("}")[-1]
                if k in names:
                    if c.attrib.get("href"): return c.attrib["href"]
                    if c.text and c.text.strip(): return c.text.strip()
            return ""
        link = f("link", "guid")
        if link.startswith("http"):
            out.append({"title": clean(f("title")), "link": link, "desc": clean(f("description", "summary", "content")), "date": f("pubDate", "published", "updated", "date")})
    return out

def ai(source, title, desc):
    prompt = (
        "You are a news editor. Using ONLY facts in the notes, write an original English news brief. "
        "Use your own wording; do not copy phrases; add no facts.\n"
        'Return JSON: {"headline": "max 90 chars, neutral", "summary": "2-3 sentences, max 60 words", '
        f'"category": one of {CATS}, "skip": true if this is not a real news item (ad, quiz, podcast promo, no facts)}}\n'
        f"Notes from {source}:\nTitle: {title}\nDescription: {desc[:1200]}")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    for attempt in range(2):
        try:
            r = json.loads(get(url, 60, body, {"Content-Type": "application/json", "x-goog-api-key": KEY}))
            return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0: time.sleep(35); continue
            raise

def similar(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.82

def main():
    cfg = json.load(open(CFG)); st = cfg.get("settings", {})
    arts = json.load(open(OUT))
    status = {"last_run": datetime.now(timezone.utc).isoformat(), "sources": {}, "added": 0, "paused": bool(st.get("paused"))}
    if st.get("paused") or not KEY:
        status["note"] = "paused" if st.get("paused") else "GEMINI_API_KEY missing"
        json.dump(status, open(STATUS, "w"), indent=1); print(status["note"]); return
    seen = {a["source_url"] for a in arts}
    titles = [a["source_title"] for a in arts[:300]] + [a["headline"] for a in arts[:300]]
    cutoff = datetime.now(timezone.utc) - timedelta(days=st.get("max_age_days", 3))
    cands = []
    for s in cfg["sources"]:
        if not s.get("enabled"): continue
        try:
            items = parse_feed(get(s["url"]))
            status["sources"][s["name"]] = {"ok": True, "items": len(items)}
            for i in items:
                i["source"] = s["name"]; i["published"] = iso(i["date"]); cands.append(i)
        except Exception as e:
            status["sources"][s["name"]] = {"ok": False, "error": str(e)[:120]}
    cands = [c for c in cands if c["link"] not in seen and c["title"] and len(c["desc"]) >= 40
             and datetime.fromisoformat(c["published"]) > cutoff]
    cands.sort(key=lambda c: c["published"], reverse=True)
    added = []
    for c in cands:
        if len(added) >= st.get("max_new_per_run", 8): break
        if any(similar(c["title"], t) for t in titles): continue
        try:
            r = ai(c["source"], c["title"], c["desc"])
        except Exception as e:
            print("AI error:", e); time.sleep(8); continue
        time.sleep(7)
        if r.get("skip") or not r.get("headline") or not r.get("summary"): continue
        if any(similar(r["headline"], t) for t in titles): continue
        cat = r.get("category") if r.get("category") in CATS else "World"
        a = {"id": hashlib.sha1(c["link"].encode()).hexdigest()[:12], "headline": r["headline"].strip(),
             "summary": r["summary"].strip(), "category": cat, "source": c["source"], "source_url": c["link"],
             "source_title": c["title"], "published": c["published"],
             "added": datetime.now(timezone.utc).isoformat(), "hidden": False}
        added.append(a); titles += [c["title"], a["headline"]]
    arts = sorted(added + arts, key=lambda a: a["published"], reverse=True)[: st.get("keep", 600)]
    json.dump(arts, open(OUT, "w"), indent=1, ensure_ascii=False)
    status["added"] = len(added)
    json.dump(status, open(STATUS, "w"), indent=1)
    print(f"Added {len(added)} articles")

if __name__ == "__main__":
    main()
