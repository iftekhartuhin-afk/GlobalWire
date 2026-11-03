#!/usr/bin/env python3
"""RSS facts -> original AI-written news article -> data/articles.json.
Uses ONLY the feed's own title + description as fact notes. Never scrapes article pages."""
import json, os, re, urllib.parse, time, html, hashlib, difflib, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/runs.json")
CFG, OUT, STATUS = (os.path.join(ROOT, p) for p in ("config/sources.json", "data/articles.json", "data/status.json"))
CATS = ["World", "Politics", "Business", "Technology", "Science", "Health", "Environment", "Sports", "Culture"]
KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ = os.environ.get("GROQ_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "")
UA = "NewsAggregatorBot/1.0 (fact-based news writer)"
STOP = set("the and for with from that this have has will after over amid into than about says said their its new more were was are not but who how what when why".split())

def get(url, timeout=25, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(t or ""))).strip()

def iso(s):
    try: d = parsedate_to_datetime(s)
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

def pick_model():
    try:
        r = json.loads(get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", 30, None, {"x-goog-api-key": KEY}))
        names = [m["name"].split("/")[-1] for m in r.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:
        print("ListModels error:", e); return MODEL
    if "gemini-flash-latest" in names: return "gemini-flash-latest"
    ok = [n for n in names if n.startswith("gemini") and "flash" in n and not any(x in n for x in ("image", "tts", "live", "preview", "exp", "thinking", "audio"))]
    return sorted(ok)[-1] if ok else MODEL

def pick_groq():
    global GROQ_MODEL
    if GROQ_MODEL: return
    try:
        r = json.loads(get("https://api.groq.com/openai/v1/models", 30, None, {"Authorization": "Bearer " + GROQ}))
        names = [m["id"] for m in r.get("data", []) if m.get("active", True)]
    except Exception as e:
        print("Groq ListModels error:", e); names = []
    for pref in ("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"):
        if pref in names: GROQ_MODEL = pref; return
    ok = [n for n in names if not any(x in n for x in ("whisper", "guard", "tts", "orpheus", "playai", "embed", "compound"))]
    GROQ_MODEL = ok[0] if ok else "llama-3.3-70b-versatile"

def ai_groq(prompt):
    body = json.dumps({"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}, "temperature": 0.4}).encode()
    r = json.loads(get("https://api.groq.com/openai/v1/chat/completions", 45, body,
                       {"Content-Type": "application/json", "Authorization": "Bearer " + GROQ}))
    return json.loads(r["choices"][0]["message"]["content"])

def ai(prompt):
    if GROQ:
        try: return ai_groq(prompt)
        except Exception as e: print("Groq error:", e)
    if not KEY: raise RuntimeError("Groq failed and no Gemini key")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}).encode()
    last = None
    for m in dict.fromkeys([MODEL, "gemini-2.5-flash", "gemini-flash-lite-latest"]):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        for attempt in range(2):
            try:
                r = json.loads(get(url, 30, body, {"Content-Type": "application/json", "x-goog-api-key": KEY}))
                return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504): time.sleep(20 if attempt == 0 else 5); continue
                break
    raise last

def words(t): return {w for w in re.findall(r"[a-z0-9]{4,}", t.lower()) if w not in STOP}

def cluster(items):
    groups = []
    for it in items:
        it["w"] = words(it["title"] + " " + it["desc"][:250])
        for g in groups:
            u = g[0]["w"] | it["w"]
            if u and len(g[0]["w"] & it["w"]) / len(u) >= 0.3 and all(x["source"] != it["source"] for x in g):
                g.append(it); break
        else: groups.append([it])
    return groups

def make_prompt(group):
    notes = "\n\n".join(f"[{i['source']}] Title: {i['title']}\nDetails: {i['desc'][:1500]}" for i in group)
    return ("You are a news reporter writing an ORIGINAL article for an international news site, using the fact notes below.\n"
            "Rules: use only facts stated in the notes; add NO details, quotes, numbers, names or background that are not in them; "
            "write in your own wording and structure, never copying phrases; neutral tone; no first person; never claim to have been on the scene or to have interviewed anyone. "
            "Attribute facts to the outlet that reported them when it matters (e.g. 'according to DW'). "
            "The lead answers who/what/when/where; add why-it-matters only if the notes say so. "
            "If the notes are thin, write a short piece (80-150 words); with richer notes up to 350 words, in 2-5 paragraphs. "
            "Also write a fluent, natural Bengali (বাংলা) version of headline, dek and body, as a Bengali news reporter would, with exactly the same facts.\n"
            f'Return JSON: {{"headline": "max 90 chars", "dek": "one sentence, max 30 words", "body": ["paragraph", "..."], "headline_bn": "Bengali headline", "dek_bn": "Bengali one-sentence summary", "body_bn": ["Bengali paragraph", "..."], "image_prompt": "max 25 words, English, a generic illustrative scene of the topic; NO real people, faces, flags, logos or text", "category": one of {CATS}, '
            '"skip": true if this is not a real news story (ad, quiz, podcast promo, no facts)}\n\nNOTES:\n' + notes)

def log_run(status):
    try: runs = json.load(open(RUNS))
    except Exception: runs = []
    runs.append({"t": status["last_run"], "added": status["added"], "note": status.get("note", "")})
    json.dump(runs[-1000:], open(RUNS, "w"))

def paras(b):
    if isinstance(b, str): b = b.split("\n")
    return [str(p).strip() for p in (b or []) if str(p).strip()]

IMG_DIR = os.path.join(ROOT, "data", "img")

def add_images(added, st, status):
    if not st.get("images", True): return
    fails = made = 0
    for a in added:
        if not a.get("image_prompt") or fails >= 2: continue
        q = urllib.parse.quote(a["image_prompt"] + ", flat vector editorial illustration, no text, no logos")
        url = f"https://image.pollinations.ai/prompt/{q}?width=800&height=450&nologo=true&model=flux&seed={int(a['id'], 16) % 100000}"
        try:
            data = get(url, 90)
            ext = "jpg" if data[:2] == b"\xff\xd8" else "png" if data[:4] == b"\x89PNG" else None
            if not ext or len(data) < 5000: raise ValueError("not an image")
            os.makedirs(IMG_DIR, exist_ok=True)
            open(os.path.join(IMG_DIR, f"{a['id']}.{ext}"), "wb").write(data)
            a["image"] = f"data/img/{a['id']}.{ext}"; made += 1; fails = 0
        except Exception as e:
            print("Image error:", e); fails += 1
        time.sleep(16)
    status["images"] = made

def prune_images(arts):
    keep = {os.path.basename(a["image"]) for a in arts if a.get("image")}
    if os.path.isdir(IMG_DIR):
        for f in os.listdir(IMG_DIR):
            if f not in keep: os.remove(os.path.join(IMG_DIR, f))

def fb_post(added, status, st):
    page, tok = os.environ.get("FB_PAGE_ID", ""), os.environ.get("FB_PAGE_TOKEN", "")
    if not (page and tok and added): return
    repo = os.environ.get("GITHUB_REPOSITORY", "/").split("/")
    base = st.get("site_url") or f"https://{repo[0]}.github.io/{repo[1]}/"
    posted = 0
    for a in added[: st.get("facebook_per_run", 2)]:
        msg = f"{a.get('headline_bn') or a['headline']}\n\n{a.get('summary_bn') or a['summary']}\n\nFacts reported by: {a['source']}"
        data = urllib.parse.urlencode({"message": msg, "link": base + "#/a/" + a["id"], "access_token": tok}).encode()
        try:
            get(f"https://graph.facebook.com/v22.0/{page}/feed", 30, data); a["fb"] = True; posted += 1
        except urllib.error.HTTPError as e:
            status["facebook"] = "error " + str(e.code) + " " + e.read().decode()[:150]; break
        except Exception as e:
            status["facebook"] = "error " + str(e)[:150]; break
        time.sleep(3)
    status.setdefault("facebook", f"posted {posted}")

def similar(a, b): return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.82

def main():
    global MODEL
    cfg = json.load(open(CFG)); st = cfg.get("settings", {})
    arts = json.load(open(OUT))
    status = {"last_run": datetime.now(timezone.utc).isoformat(), "sources": {}, "added": 0, "paused": bool(st.get("paused"))}
    if st.get("paused") or not (KEY or GROQ):
        status["note"] = "paused" if st.get("paused") else "No AI key set (GROQ_API_KEY or GEMINI_API_KEY)"
        json.dump(status, open(STATUS, "w"), indent=1); log_run(status); print(status["note"]); return
    if KEY: MODEL = pick_model()
    if GROQ: pick_groq()
    print("Gemini:", MODEL, "| Groq:", GROQ_MODEL if GROQ else "off")
    seen = {a["source_url"] for a in arts} | {s["url"] for a in arts for s in a.get("sources", [])}
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
             and datetime.fromisoformat(c["published"]) > cutoff and not any(similar(c["title"], t) for t in titles)]
    cands.sort(key=lambda c: c["published"], reverse=True)
    groups = cluster(cands)
    groups.sort(key=lambda g: (-len(g), -max(datetime.fromisoformat(i["published"]).timestamp() for i in g)))
    added = []; errors = 0
    for g in groups:
        if len(added) >= st.get("max_new_per_run", 8): break
        try:
            r = ai(make_prompt(g)); errors = 0
        except Exception as e:
            print("AI error:", e); errors += 1
            if errors >= 3: status["note"] = "AI error: " + str(e)[:150]; break
            time.sleep(8); continue
        time.sleep(5)
        body = r.get("body")
        if isinstance(body, str): body = [p.strip() for p in body.split("\n") if p.strip()]
        if r.get("skip") or not r.get("headline") or not body or not isinstance(body, list): continue
        if any(similar(r["headline"], t) for t in titles): continue
        a = {"id": hashlib.sha1(g[0]["link"].encode()).hexdigest()[:12], "headline": r["headline"].strip(),
             "summary": (r.get("dek") or "").strip(), "body": [str(p).strip() for p in body if str(p).strip()],
             "headline_bn": str(r.get("headline_bn") or "").strip(), "summary_bn": str(r.get("dek_bn") or "").strip(),
             "body_bn": paras(r.get("body_bn")), "image_prompt": str(r.get("image_prompt") or "")[:300],
             "category": r.get("category") if r.get("category") in CATS else "World",
             "source": ", ".join(i["source"] for i in g), "source_url": g[0]["link"], "source_title": g[0]["title"],
             "sources": [{"name": i["source"], "url": i["link"], "title": i["title"]} for i in g],
             "published": max(i["published"] for i in g), "added": datetime.now(timezone.utc).isoformat(), "hidden": False}
        added.append(a); titles += [a["headline"]] + [i["title"] for i in g]
    add_images(added, st, status); fb_post(added, status, st)
    arts = sorted(added + arts, key=lambda a: a["published"], reverse=True)[: st.get("keep", 600)]
    prune_images(arts)
    json.dump(arts, open(OUT, "w"), indent=1, ensure_ascii=False)
    status["added"] = len(added)
    json.dump(status, open(STATUS, "w"), indent=1); log_run(status)
    print(f"Added {len(added)} articles")

if __name__ == "__main__":
    main()
