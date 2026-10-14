"""Bounded, interactive collection from ordinary browser pages.

No imported cookies, saved browser profile, CDP port, signed-request recreation,
CAPTCHA solver, proxy rotation, or synthetic collection output.  Browser response
parsers are passive adapters for data the open page itself receives, not API clients.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import queue
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit, parse_qs
from zoneinfo import ZoneInfo

PLATFORMS = ("小红书", "抖音", "微博", "B站")
TZ = ZoneInfo("Asia/Shanghai")
FIELDS = [
    "platform", "query", "record_type", "topic_id", "topic_url", "topic_title", "topic_text",
    "topic_time_raw", "topic_time_iso", "topic_time_status", "comment_id", "comment_id_status",
    "comment_id_evidence", "local_record_id", "dedup_basis", "parent_comment_id", "text",
    "comment_time_raw", "comment_time_iso", "comment_time_status", "scraped_at", "source_url",
    "extraction_method", "text_status", "coverage",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_root(start=None):
    """Resolve from python/ or project root; never depend on a user's absolute path."""
    start = Path(start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        if (p / "python").is_dir() and (p / "数据").is_dir():
            return p
    # The shipped helper's own location is an unambiguous fallback.
    p = Path(__file__).resolve().parent.parent
    if (p / "python").is_dir() and (p / "数据").is_dir():
        return p
    raise FileNotFoundError("找不到项目根目录。请保留 lasso论文/数据 和 lasso论文/python 的目录结构。")


def output_dir(platform, root=None):
    if platform not in PLATFORMS:
        raise ValueError("未知平台")
    base = (Path(root).resolve() if root else project_root()) / "数据" / "01_原始数据" / platform / "new"
    base.mkdir(parents=True, exist_ok=True)
    return base


def parse_time(value, captured_at=None):
    """Return ISO + provenance. Never silently assign a year to an MM-DD label."""
    raw = "" if value is None else str(value).strip()
    if not raw:
        return "", "missing"
    anchor = datetime.fromisoformat(captured_at.replace("Z", "+00:00")) if captured_at else datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    anchor = anchor.astimezone(TZ)
    if re.fullmatch(r"\d{10}|\d{13}", raw):
        val = int(raw) / (1000 if len(raw) == 13 else 1)
        try:
            return datetime.fromtimestamp(val, TZ).isoformat(timespec="seconds"), "unix_timestamp"
        except (OverflowError, ValueError, OSError):
            return "", "unparseable"
    s = re.sub(r"^(?:发布于|发表于|编辑于|更新于|发布时间[:：]?)[\s]*", "", raw)
    # Preserve the original value; this cleaning applies only to the parsed column.
    m = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", s)
    if not m:
        m = re.search(r"\d{4}年\d{1,2}月\d{1,2}日(?:\s*\d{1,2}:\d{2}(?::\d{2})?)?", s)
    if m:
        t = m.group().replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").strip()
        date_only = ":" not in t
        try:
            # strptime also handles dates without zero padding.
            d = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            try:
                d = datetime.strptime(t, "%Y-%m-%d %H:%M" if ":" in t else "%Y-%m-%d")
            except ValueError:
                return "", "unparseable"
        if d.tzinfo is None:
            d = d.replace(tzinfo=TZ)
        return (d.date().isoformat() if date_only else d.isoformat(timespec="seconds")), ("absolute_date" if date_only else "absolute_datetime")
    try:
        d = parsedate_to_datetime(s)
        if d.tzinfo is not None:
            return d.astimezone(TZ).isoformat(timespec="seconds"), "absolute_datetime"
    except (ValueError, TypeError):
        pass
    m = re.match(r"^(\d+)\s*(秒|分钟|小时|天)前", s)
    if m:
        unit = {"秒": 1, "分钟": 60, "小时": 3600, "天": 86400}[m[2]]
        return (anchor - timedelta(seconds=int(m[1]) * unit)).isoformat(timespec="seconds"), "relative_estimate"
    m = re.match(r"^(今天|昨天|前天)(?:\s*(\d{1,2}):(\d{2}))?", s)
    if m:
        d = anchor - timedelta(days={"今天": 0, "昨天": 1, "前天": 2}[m[1]])
        if m[2]:
            return d.replace(hour=int(m[2]), minute=int(m[3]), second=0, microsecond=0).isoformat(), "relative_day_inferred"
        return d.date().isoformat(), "relative_day_inferred"
    if re.match(r"^\d+\s*(个月|月|年)前", s):
        return "", "relative_coarse_unresolved"
    if s == "刚刚":
        return anchor.isoformat(timespec="seconds"), "relative_estimate"
    if re.search(r"(?<!\d)\d{1,2}[-月]\d{1,2}(?:日)?", s):
        return "", "missing_year"
    return "", "unparseable"


def search_url(platform, query, page=1):
    if platform == "小红书":
        return "https://www.xiaohongshu.com/search_result?" + urlencode({"keyword": query})
    if platform == "抖音":
        return "https://www.douyin.com/search/" + quote(query, safe="") + "?type=video"
    if platform == "微博":
        return "https://s.weibo.com/weibo?" + urlencode({"q": query, "page": page})
    if platform == "B站":
        return "https://search.bilibili.com/video?" + urlencode({"keyword": query, "page": page})
    raise ValueError(platform)


def topic_id_from_url(platform, url):
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    expected = {"小红书": "xiaohongshu.com", "抖音": "douyin.com", "微博": "weibo.com", "B站": "bilibili.com"}[platform]
    if host != expected and not host.endswith("." + expected):
        return ""
    patterns = {"小红书": r"/(?:explore|search_result)/([0-9a-f]{24})(?:/|$)", "抖音": r"/(?:video|note)/(\d+)(?:/|$)", "B站": r"/video/(BV[0-9A-Za-z]+)(?:/|$)", "微博": r"/(?:\d+|detail)/([A-Za-z0-9]+)(?:/|$)"}
    m = re.search(patterns[platform], parts.path)
    return m[1] if m else ""


def canonical_url(platform, topic_id, observed_url=""):
    if platform == "小红书": return f"https://www.xiaohongshu.com/explore/{topic_id}"
    if platform == "抖音":
        kind = "note" if topic_id_from_url(platform, observed_url) == topic_id and "/note/" in urlsplit(observed_url).path else "video"
        return f"https://www.douyin.com/{kind}/{topic_id}"
    if platform == "B站": return f"https://www.bilibili.com/video/{topic_id}"
    if observed_url and topic_id_from_url(platform, observed_url):
        return "https://weibo.com" + urlsplit(observed_url).path
    return f"https://weibo.com/detail/{topic_id}"


@dataclass
class CrawlConfig:
    platform: str
    queries: list[str]
    root: str | None = None
    max_topics: int = 5
    max_comments_per_topic: int = 30
    search_scrolls: int = 3
    comment_scrolls: int = 6
    search_pages: int = 2
    pause_seconds: float = 3.0
    max_runtime_minutes: float = 20.0
    interactive: bool = True
    browser_channel: str | None = None

    def validate(self):
        if self.platform not in PLATFORMS: raise ValueError("平台名错误")
        if not self.queries or any(not isinstance(q, str) or not q.strip() for q in self.queries):
            raise ValueError("QUERIES必须是非空的搜索词字符串列表")
        if not 1 <= self.max_topics <= 100: raise ValueError("MAX_TOPICS范围为1–100")
        if not 1 <= self.max_comments_per_topic <= 500: raise ValueError("MAX_COMMENTS_PER_TOPIC范围为1–500")
        if not 0 <= self.search_scrolls <= 30 or not 0 <= self.comment_scrolls <= 50: raise ValueError("滚动窗口过大")
        if not 1 <= self.search_pages <= 10: raise ValueError("SEARCH_PAGES范围为1–10")
        if self.pause_seconds < 3: raise ValueError("为避免密集请求，PAUSE_SECONDS不得少于3秒")
        if not 1 <= self.max_runtime_minutes <= 120: raise ValueError("单次运行时间范围为1–120分钟")
        if self.browser_channel not in (None, "chrome", "msedge"): raise ValueError("仅支持已安装的Chromium、Chrome或Edge")


def make_record(platform, query, topic, item, method, occurrence=0):
    scraped_at = item.get("scraped_at") or now_iso()
    native = str(item.get("comment_id") or "")
    record_type = item.get("record_type", "comment")
    text = str(item.get("text") or "").strip()
    if not text or not topic.get("id"):
        return None
    traw = str(topic.get("time_raw") or "")
    craw = str(item.get("time_raw") or "")
    tiso, ts = parse_time(traw, scraped_at)
    ciso, cs = parse_time(craw, scraped_at)
    if record_type == "post":
        status, basis, identity = "not_applicable_post", "native_post_id", str(topic["id"])
    elif native:
        status, basis, identity = "native", "native_comment_id", native
    else:
        status, basis = "missing", "local_text_time_occurrence"
        identity = json.dumps([text, craw, occurrence], ensure_ascii=False)
    local_id = "local_" + hashlib.sha256((platform + "|" + str(topic["id"]) + "|" + basis + "|" + identity).encode()).hexdigest()[:24]
    url = canonical_url(platform, str(topic["id"]), topic.get("url", ""))
    return dict(zip(FIELDS, [
        platform, query, record_type, str(topic["id"]), url, topic.get("title", ""), topic.get("text", ""),
        traw, tiso, ts, native, status, item.get("id_evidence", "") if native else "", local_id, basis,
        str(item.get("parent_comment_id") or ""), text, craw, ciso, cs, scraped_at, url, method,
        item.get("text_status", "as_delivered"), "bounded_page_window",
    ]))


class RecordStore:
    """SQLite transactions are the checkpoint; CSV/JSONL are refreshable exports in new/."""
    def __init__(self, config, query):
        self.config, self.query = config, query
        limits = {k: v for k, v in asdict(config).items() if k not in ("root", "interactive", "browser_channel", "queries", "pause_seconds", "max_runtime_minutes")}
        signature = hashlib.sha256(json.dumps([query, limits], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
        slug = re.sub(r'[^\w\-\u4e00-\u9fff]+', '_', query).strip('_')[:35] or "query"
        self.prefix = output_dir(config.platform, config.root) / f"{slug}_{signature}"
        self.db = sqlite3.connect(str(self.prefix) + ".sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (local_id TEXT PRIMARY KEY, topic_id TEXT NOT NULL, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS topics (topic_id TEXT PRIMARY KEY, status TEXT NOT NULL, detail TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("query", query))
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("limits", json.dumps(limits, ensure_ascii=False)))
        self.db.commit()

    def add(self, record):
        if record is None: return False
        cur = self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?)", (record["local_record_id"], record["topic_id"], json.dumps(record, ensure_ascii=False)))
        self.db.commit()
        return bool(cur.rowcount)

    def count(self, topic_id=None):
        sql = "SELECT COUNT(*) FROM records" + (" WHERE topic_id=?" if topic_id else "")
        return self.db.execute(sql, (str(topic_id),) if topic_id else ()).fetchone()[0]

    def mode(self, topic_id):
        row = self.db.execute("SELECT payload FROM records WHERE topic_id=? ORDER BY rowid LIMIT 1", (str(topic_id),)).fetchone()
        return json.loads(row[0]).get("extraction_method") if row else None

    def incomplete_topics(self):
        return self.db.execute("SELECT COUNT(*) FROM topics WHERE status != 'window_finished'").fetchone()[0]

    def done(self, topic_id):
        row = self.db.execute("SELECT status FROM topics WHERE topic_id=?", (str(topic_id),)).fetchone()
        return bool(row and row[0] == "window_finished")

    def mark(self, topic_id, status, detail=""):
        self.db.execute("INSERT OR REPLACE INTO topics VALUES (?,?,?)", (str(topic_id), status, detail))
        self.db.commit()

    def export(self):
        rows = [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM records ORDER BY rowid")]
        result = {"query": self.query, "records": len(rows), "csv": None, "jsonl": None, "checkpoint": str(self.prefix) + ".sqlite3", "export_errors": []}
        for suffix in (".csv", ".jsonl"):
            dest = Path(str(self.prefix) + suffix)
            tmp = dest.with_suffix(suffix + ".tmp")
            try:
                with tmp.open("w", encoding="utf-8-sig" if suffix == ".csv" else "utf-8", newline="") as f:
                    if suffix == ".csv":
                        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
                    else:
                        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
                tmp.replace(dest)
                result[suffix[1:]] = str(dest)
            except OSError as e:
                result["export_errors"].append(suffix + ":" + type(e).__name__)
        return result

    def close(self):
        self.db.close()


# Only selected public content fields leave the page. No author profiles or cookies.
SEARCH_LINKS_JS = r"""platform => {
 const patterns = {'小红书': /\/(?:explore|search_result)\/([0-9a-f]{24})(?:\?|\/|$)/,
 '抖音': /\/(?:video|note)\/(\d+)(?:\?|\/|$)/, 'B站': /\/video\/(BV[0-9A-Za-z]+)(?:\?|\/|$)/};
 const pattern=patterns[platform], found=[], seen=new Set();
 for(const a of document.querySelectorAll('a[href]')) {
   const m=a.href.match(pattern); if(!m || seen.has(m[1])) continue;
   const r=a.getBoundingClientRect(); if(r.width<=0 || r.height<=0) continue;
   const card=platform==='小红书' ? a.closest('section.note-item,.note-item') : platform==='B站' ? a.closest('.bili-video-card,.video-list-item') : a;
   if(!card)continue; // Exclude navigation/footer links, including B站's 高级弹幕 tutorial.
   seen.add(m[1]);
   const t=card.querySelector('.title,.bili-video-card__info--tit,h3,.VDYK8Xd7');
   found.push({id:m[1],url:a.href,title:(t?.innerText || a.title || '').trim(),text:'',time_raw:''});
 }
 return found;
}"""

TOPIC_META_JS = r"""platform => {
 const txt=s=>document.querySelector(s)?.innerText?.trim() || '';
 if(platform==='小红书') return {title:txt('#detail-title'), text:txt('#detail-desc'), time_raw:txt('.note-scroller .date,.note-content .date,.note-container .date')};
 if(platform==='B站') { const p=document.querySelector('bili-comments')?.getAttribute('data-params')||'';
   return {title:txt('h1.video-title,h1'),text:txt('.basic-desc-info'),time_raw:txt('.pubdate-ip-text,.pubdate-text'),aid:/^1,\d+$/.test(p)?p.split(',')[1]:''}; }
 return {title:txt('[data-e2e="video-desc"],h1'),text:txt('[data-e2e="video-desc"],h1'),time_raw:txt('[data-e2e="detail-video-publish-time"],[data-e2e="video-create-time"]')};
}"""

XHS_COMMENTS_JS = r"""() => {
 const contentText = root => { let out='';
   function walk(n){if(n.nodeType===3){out+=n.textContent;return;}
     if(n.nodeType!==1 || ['STYLE','SCRIPT','NOSCRIPT','TEMPLATE'].includes(n.tagName))return;
     if(n.tagName==='IMG'){out+=n.getAttribute('alt')||'';return;} n.childNodes.forEach(walk);}
   if(root)walk(root);return out.trim();};
 return [...document.querySelectorAll('[id^="comment-"]')].map(el=>{
   const id=/^comment-([0-9a-f]{24})$/.exec(el.id);
   const text=contentText(el.querySelector('.content,.note-text'));
   const date=el.querySelector('.date');
   return {comment_id:id?id[1]:'',id_evidence:id?'DOM id='+el.id:'',text,
     time_raw:date?.getAttribute('title') || date?.innerText?.trim() || '',text_status:'visible_dom'};
 }).filter(x=>x.text);
}"""

BILI_COMMENTS_JS = r"""() => {
 function all(root, selector) { const found=[...root.querySelectorAll(selector)];
   for(const el of root.querySelectorAll('*')) if(el.shadowRoot) found.push(...all(el.shadowRoot,selector)); return found; }
 function text(root) { let out=''; function walk(n){
   if(n.nodeType===3){out+=n.textContent;return;}
   if(n.nodeType!==1 && n.nodeType!==11)return;
   if(['STYLE','SCRIPT','NOSCRIPT','TEMPLATE'].includes(n.tagName))return;
   if(n.tagName==='IMG'){out+=n.getAttribute('alt')||'';return;}
   if(n.shadowRoot)walk(n.shadowRoot);else n.childNodes.forEach(walk);
 } if(root)walk(root);return out.trim();}
 return all(document,'bili-comment-renderer,bili-comment-reply-renderer').map(el=>{
   const root=el.shadowRoot || el;
   const rich=all(root,'bili-rich-text')[0], date=all(root,'#pubdate')[0];
   const native=el.getAttribute('data-rpid') || '';
   return {comment_id:/^\d+$/.test(native)?native:'',id_evidence:/^\d+$/.test(native)?'DOM data-rpid':'',
      text:text(rich),time_raw:date?.getAttribute('title') || text(date),text_status:'visible_dom'};
 }).filter(x=>x.text);
}"""

DOUYIN_COMMENTS_JS = r"""() => {
 const text = root => { let out=''; function walk(n){
   if(n.nodeType===3){out+=n.textContent;return;}
   if(n.nodeType!==1 || ['SCRIPT','STYLE','NOSCRIPT','TEMPLATE'].includes(n.tagName))return;
   if(n.tagName==='IMG'){out+=n.getAttribute('alt')||'';return;}n.childNodes.forEach(walk);
 }if(root)walk(root);return out.trim();};
 // These two classes were verified on the live public DOM on 2026-10-14.
 // If a site release changes them, fail empty rather than extract a whole user/profile card.
 return [...document.querySelectorAll('[data-e2e="comment-item"]')].map(el=>({
    comment_id:'',id_evidence:'',text:text(el.querySelector('.FduGc_lz')),
    time_raw:text(el.querySelector('.VAQA49VP > span')),text_status:'visible_dom'
 })).filter(x=>x.text);
}"""


WEIBO_POSTS_JS = r"""() => {
 function content(root){let out='';function walk(n){
   if(n.nodeType===3){out+=n.textContent;return;}
   if(n.nodeType!==1 || ['SCRIPT','STYLE','NOSCRIPT','TEMPLATE'].includes(n.tagName))return;
   if(['fl_fold','fl_unfold'].includes(n.getAttribute('action-type')))return;
   if(n.tagName==='IMG'){out+=n.getAttribute('alt')||'';return;}
   if(n.tagName==='BR'){out+='\n';return;}n.childNodes.forEach(walk);
 }if(root)walk(root);return out.trim();}
 return [...document.querySelectorAll('.card-wrap[mid]')].map(card=>{
 const full=card.querySelector('[node-type="feed_list_content_full"]');
 const body=full || card.querySelector('[node-type="feed_list_content"]');
 const date=card.querySelector('.from a[href*="weibo.com"],.from a[href^="//"]');
 const shortened=!!body?.querySelector('[action-type="fl_unfold"]') || /展开/.test(body?.querySelector('a:last-child')?.textContent||'');
 return {id:card.getAttribute('mid'),url:date?.href||'',title:'',text:content(body),
 time_raw:date?.getAttribute('title')||date?.innerText?.trim()||'',text_status:full?'full_dom':(shortened?'possibly_truncated':'visible_dom')};
}).filter(x=>/^\d+$/.test(x.id||'') && x.text);
}"""


def _clean_text(value):
    return html.unescape(re.sub(r"<[^>]+>", "", str(value or ""))).strip()


def response_items(platform, url, data, topic):
    """Pure parser. Reject a response whose native topic ID is not verified to match."""
    if not isinstance(data, dict): return []
    qs = parse_qs(urlsplit(url).query)
    result = []
    if platform == "小红书" and "/comment/" in url:
        requested = (qs.get("note_id") or [""])[0]
        if requested != str(topic.get("id")): return []
        comments = (data.get("data") or {}).get("comments", [])
        def xhs(rows, parent=""):
            for c in rows or []:
                if not isinstance(c, dict): continue
                if c.get("note_id") and str(c["note_id"]) != requested: continue
                result.append({"comment_id":str(c.get("id") or ""), "id_evidence":"page_response:comment.id + request.note_id", "text":c.get("content", ""), "time_raw":c.get("create_time", ""), "parent_comment_id":parent})
                xhs(c.get("sub_comments", []), str(c.get("id") or ""))
        xhs(comments)
    elif platform == "抖音" and "/comment/" in url:
        requested = (qs.get("aweme_id") or [""])[0]
        if requested != str(topic.get("id")): return []
        for c in data.get("comments") or []:
            if not isinstance(c, dict): continue
            if c.get("aweme_id") and str(c["aweme_id"]) != requested: continue
            result.append({"comment_id":str(c.get("cid") or ""), "id_evidence":"page_response:comment.cid + request.aweme_id", "text":c.get("text", ""), "time_raw":c.get("create_time", ""), "parent_comment_id":"" if str(c.get("reply_id") or "") == "0" else str(c.get("reply_id") or "")})
    elif platform == "B站" and "/reply/" in url:
        requested = (qs.get("oid") or [""])[0]
        if not topic.get("aid") or requested != str(topic["aid"]): return []
        d = data.get("data") or {}
        candidates = list(d.get("replies") or []) + list(d.get("top_replies") or [])
        top = d.get("top") or {}
        if isinstance(top, dict): candidates += [v for v in top.values() if isinstance(v, dict)]
        def bili(rows, parent=""):
            for c in rows:
                if not isinstance(c, dict): continue
                if c.get("oid") and str(c["oid"]) != requested: continue
                result.append({"comment_id":str(c.get("rpid_str") or c.get("rpid") or ""), "id_evidence":"page_response:rpid + verified video aid", "text":(c.get("content") or {}).get("message", ""), "time_raw":c.get("ctime", ""), "parent_comment_id":parent})
                bili(c.get("replies") or [], str(c.get("rpid_str") or c.get("rpid") or ""))
        bili(candidates)
    return result


def search_topics_from_response(platform, url, data, query, request_keyword=""):
    """Search response adapter. Keyword equality is required; never use a feed."""
    if not isinstance(data, dict): return []
    qs = parse_qs(urlsplit(url).query)
    keyword = (qs.get("keyword") or [request_keyword])[0]
    if keyword != query: return []
    result = []
    if platform == "小红书" and "/search/notes" in url:
        for item in (data.get("data") or {}).get("items") or []:
            card = item.get("note_card") or {}
            ident = str(item.get("id") or card.get("note_id") or "")
            if not re.fullmatch(r"[a-f0-9]{24}", ident): continue
            link = canonical_url(platform, ident)
            # A page-provided navigation token stays in memory only, never in exports/checkpoints.
            nav = item.get("xsec_token") or card.get("xsec_token")
            if nav: link += "?" + urlencode({"xsec_token":nav, "xsec_source":"pc_search"})
            result.append({"id":ident,"url":link,"title":card.get("display_title") or card.get("title") or "", "text":card.get("desc") or "", "time_raw":card.get("time") or ""})
    elif platform == "抖音" and "/search/" in url:
        candidates = [x.get("aweme_info") or {} for x in data.get("data") or [] if isinstance(x, dict)]
        candidates += data.get("aweme_list") or []
        for info in candidates:
            ident = str(info.get("aweme_id") or "")
            if not re.fullmatch(r"\d+", ident): continue
            result.append({"id":ident,"url":canonical_url(platform, ident),"title":info.get("desc") or "", "text":info.get("desc") or "", "time_raw":info.get("create_time") or ""})
    elif platform == "B站" and "/search/type" in url and (qs.get("search_type") or [""])[0] == "video":
        for info in (data.get("data") or {}).get("result") or []:
            ident = str(info.get("bvid") or "")
            if not re.fullmatch(r"BV[0-9A-Za-z]+", ident): continue
            result.append({"id":ident,"aid":str(info.get("aid") or ""),"url":canonical_url(platform, ident),"title":_clean_text(info.get("title")), "text":_clean_text(info.get("description")), "time_raw":info.get("pubdate") or ""})
    return result


def update_topic_from_response(platform, data, topic):
    """Update only a response object explicitly carrying the selected native topic ID."""
    if not isinstance(data, dict): return
    if platform == "小红书":
        d = data.get("data") or {}
        for item in d.get("items", []) if isinstance(d, dict) else []:
            card = item.get("note_card") or {}
            ident = str(card.get("note_id") or item.get("id") or "")
            if ident == str(topic["id"]):
                topic.update(title=card.get("title") or topic.get("title", ""), text=card.get("desc") or topic.get("text", ""), time_raw=card.get("time") or topic.get("time_raw", ""))
    elif platform == "抖音":
        info = data.get("aweme_detail") or {}
        if str(info.get("aweme_id", "")) == str(topic["id"]):
            topic.update(title=info.get("desc") or topic.get("title", ""), text=info.get("desc") or topic.get("text", ""), time_raw=info.get("create_time") or topic.get("time_raw", ""))
    elif platform == "B站":
        d = data.get("data") or {}
        candidates = [d, d.get("View", {})] if isinstance(d, dict) else []
        for info in candidates:
            if str(info.get("bvid", "")) == str(topic["id"]):
                topic.update(aid=str(info.get("aid") or ""), title=info.get("title") or topic.get("title", ""), text=info.get("desc") or topic.get("text", ""), time_raw=info.get("pubdate") or topic.get("time_raw", ""))


class SafeStop(Exception):
    pass


class BrowserCollector:
    def __init__(self, config, messages, replies, cancelled):
        self.cfg, self.messages, self.replies, self.cancelled = config, messages, replies, cancelled
        self.started = time.monotonic()
        self.topic = None
        self.responses = []
        self.search_responses = []
        self.active_query = ""
        self.buffer = []
        self.last_http_problem = ""

    def log(self, text): self.messages.put(("progress", text))

    def check_budget(self):
        if self.cancelled.is_set(): raise SafeStop("用户中止；已落盘数据和检查点保留")
        if time.monotonic() - self.started > self.cfg.max_runtime_minutes * 60:
            raise SafeStop("已到单次运行时间上限；已落盘数据和检查点保留")

    def wait(self, page):
        self.check_budget()
        page.wait_for_timeout(int(self.cfg.pause_seconds * 1000))
        self.check_budget()

    def manual(self, page, reason):
        if not self.cfg.interactive:
            raise SafeStop(reason + "；需要手动处理，未继续请求")
        self.messages.put(("input", reason + "。请在刚打开的独立浏览器中手动处理；完成后输入 r，停止输入 q："))
        while True:
            self.check_budget()
            try: answer = self.replies.get_nowait()
            except queue.Empty:
                page.wait_for_timeout(250); continue
            if answer.strip().lower() == "r": return
            raise SafeStop("用户停止；可保留检查点后重新运行")

    def on_response(self, response, search=False):
        u = response.url
        host = (urlsplit(u).hostname or "").lower()
        expected = {"小红书":"xiaohongshu.com", "抖音":"douyin.com", "微博":"weibo.com", "B站":"bilibili.com"}[self.cfg.platform]
        if not (host == expected or host.endswith("." + expected)): return
        # No request headers, request body, full response body or URLs are persisted.
        paths = {"小红书":("/comment/", "/feed"), "抖音":("/comment/", "/aweme/detail/"), "微博":(), "B站":("/reply/", "/web-interface/view", "/web-interface/wbi/view")}[self.cfg.platform]
        is_search = search and "/search/" in urlsplit(u).path
        if not is_search and not any(x in u for x in paths): return
        if response.status in (401, 403, 412, 429):
            self.last_http_problem = f"页面内容请求返回 HTTP {response.status}，可能需要登录、验证或稍后再试"
            return
        (self.search_responses if is_search else self.responses).append(response)

    def drain_search(self):
        pending, self.search_responses = self.search_responses, []
        found = []
        for response in pending:
            try:
                keyword = ""
                if response.request.method == "POST":
                    body = response.request.post_data_json
                    if isinstance(body, dict): keyword = str(body.get("keyword") or "")
                found.extend(search_topics_from_response(self.cfg.platform, response.url, response.json(), self.active_query, keyword))
            except Exception:
                continue
        return found

    def drain(self):
        # Called only on the Playwright owner thread, outside an event callback.
        pending, self.responses = self.responses, []
        parsed = []
        for response in pending:
            try:
                data = response.json()
                if self.topic:
                    update_topic_from_response(self.cfg.platform, data, self.topic)
                    parsed.append((response.url, data))
            except Exception:
                continue
        for url, data in parsed:
            self.buffer.extend(response_items(self.cfg.platform, url, data, self.topic))

    def gate_reason(self, page):
        if self.last_http_problem: return self.last_http_problem
        url = page.url.lower()
        if any(x in url for x in ("passport.weibo.com", "/login", "/captcha", "/verify")):
            return "页面进入登录或安全验证"
        # Check displayed challenge components, not comment text containing these words.
        selectors = '[class*="captcha"],[id*="captcha"],[class*="verify-wrap"],[class*="verify-dialog"],[class*="login-modal"],[class*="login-container"],[data-e2e="login-dialog"]'
        for node in page.locator(selectors).all()[:30]:
            try:
                if node.is_visible():
                    box = node.bounding_box() or {}
                    if len(node.inner_text().strip()) > 0 or (box.get("width", 0) > 100 and box.get("height", 0) > 40):
                        return "出现登录或验证窗口"
            except Exception: pass
        body = page.locator("body").inner_text(timeout=5000)
        if len(body) < 1500 and any(x in body for x in ("访问频繁", "操作频繁", "访问受限", "请完成验证", "请进行验证", "安全验证", "网络环境存在风险", "登录后查看更多", "登录后查看")):
            return "页面要求登录、验证或限制访问"
        return ""

    def ensure_ready(self, page):
        reason = self.gate_reason(page)
        if reason:
            self.manual(page, reason)
            self.last_http_problem = ""
            self.responses.clear(); self.buffer.clear()
            reason = self.gate_reason(page)
            if reason: raise SafeStop(reason + "；验证尚未解除，停止采集")

    def navigate(self, page, url):
        self.check_budget(); self.responses.clear(); self.buffer.clear(); self.last_http_problem = ""
        response = page.goto(url, wait_until="domcontentloaded", timeout=45000)
        if response and response.status in (401, 403, 412, 429):
            self.last_http_problem = f"页面返回 HTTP {response.status}"
        self.wait(page); self.ensure_ready(page)

    def collect_comments(self, page, topic, query, store):
        self.topic = topic
        self.navigate(page, topic["url"])
        if topic_id_from_url(self.cfg.platform, page.url) != str(topic["id"]):
            raise SafeStop("主题页跳转后的ID与搜索结果不一致，未采集该页评论")
        meta = page.evaluate(TOPIC_META_JS, self.cfg.platform)
        for k, v in meta.items():
            if v and (k in ("title", "text", "aid") or not topic.get(k)): topic[k] = v
        self.drain()
        mode = store.mode(topic["id"])
        for window in range(self.cfg.comment_scrolls + 1):
            self.ensure_ready(page); self.drain()
            items, self.buffer = self.buffer, []
            method = "page_response"
            # Keep one identity method per topic, including across checkpoint resumes.
            # Mixing a missing-ID DOM row with its later native-ID row would double count it.
            if mode == "visible_dom": items = []
            if not items and mode != "page_response" and self.cfg.platform in ("小红书", "抖音", "B站"):
                items = page.evaluate({"小红书":XHS_COMMENTS_JS,"抖音":DOUYIN_COMMENTS_JS,"B站":BILI_COMMENTS_JS}[self.cfg.platform])
                method = "visible_dom"
            if items and mode is None: mode = method
            collisions = {}
            for item in items:
                key = (str(item.get("text", "")), str(item.get("time_raw", "")))
                occurrence = collisions.get(key, 0); collisions[key] = occurrence + 1
                if store.count(topic["id"]) >= self.cfg.max_comments_per_topic: break
                store.add(make_record(self.cfg.platform, query, topic, item, method, occurrence))
            if store.count(topic["id"]) >= self.cfg.max_comments_per_topic: break
            if window == self.cfg.comment_scrolls: break
            # Hovering a verified visible comment region lets the normal page load the next window.
            if self.cfg.platform == "小红书":
                region = page.locator('.note-scroller,.comments-container').first
            elif self.cfg.platform == "B站":
                region = page.locator('bili-comments,#comment,.comment-container').first
            else:
                region = page.locator('[data-e2e="comment-list"],[data-e2e="comment-item"]').first
            try:
                if region.count() and region.is_visible(): region.hover(timeout=3000)
            except Exception: pass
            page.mouse.wheel(0, 650)
            self.wait(page)
        n = store.count(topic["id"])
        if n == 0:
            self.manual(page, "未读到可核验关联的评论。若评论折叠，请手动打开评论区；若有登录或验证码，请自行处理")
            self.ensure_ready(page); self.drain()
            items, self.buffer = self.buffer, []
            method = "page_response"
            if not items and self.cfg.platform in ("小红书", "抖音", "B站"):
                items = page.evaluate({"小红书":XHS_COMMENTS_JS,"抖音":DOUYIN_COMMENTS_JS,"B站":BILI_COMMENTS_JS}[self.cfg.platform])
                method = "visible_dom"
            for item in items[:self.cfg.max_comments_per_topic]: store.add(make_record(self.cfg.platform, query, topic, item, method))
            n = store.count(topic["id"])
            if n == 0:
                store.mark(topic["id"], "no_verified_comments", "未捕获可核验评论；不代表主题确实零评论")
                self.log(f"主题 {topic['id']}：没有可核验评论，保留未完成状态")
                self.topic = None; return
        store.mark(topic["id"], "window_finished", f"有限窗口结束，{n}条；非全量声明")
        self.log(f"主题 {topic['id']}：累计保存 {n} 条")
        self.topic = None

    def collect_query(self, browser, query, store):
        self.active_query = query
        self.search_responses.clear()
        search = browser.new_page(); detail = browser.new_page()
        search.on("response", lambda response: self.on_response(response, search=True))
        detail.on("response", self.on_response)
        try:
            self.navigate(search, search_url(self.cfg.platform, query))
            if self.cfg.platform == "微博":
                for page_no in range(1, self.cfg.search_pages + 1):
                    if page_no > 1: self.navigate(search, search_url("微博", query, page_no))
                    posts = search.evaluate(WEIBO_POSTS_JS)
                    if not posts:
                        self.manual(search, "搜索页未读到微博正文。请确认关键词结果已显示并完成所需登录")
                        self.ensure_ready(search); posts = search.evaluate(WEIBO_POSTS_JS)
                    if not posts: raise SafeStop("微博搜索页未读到正文；可能无结果、需登录或页面结构已变化，未生成伪数据")
                    for post in posts:
                        if store.count() >= self.cfg.max_topics: return
                        post["title"] = ""
                        item = {"record_type":"post", "text":post["text"], "time_raw":post["time_raw"], "text_status":post["text_status"]}
                        store.add(make_record("微博", query, post, item, "search_result_dom"))
                return
            topics = {}
            for window in range(self.cfg.search_scrolls + 1):
                self.ensure_ready(search)
                for t in self.drain_search() + search.evaluate(SEARCH_LINKS_JS, self.cfg.platform):
                    if topic_id_from_url(self.cfg.platform, t["url"]) == t["id"]: topics.setdefault(t["id"], t)
                if len(topics) >= self.cfg.max_topics: break
                if window < self.cfg.search_scrolls:
                    search.mouse.wheel(0, 650); self.wait(search)
            if not topics:
                self.manual(search, "搜索页未读到主题链接。请确认关键词搜索结果已显示并完成所需登录")
                self.ensure_ready(search)
                for t in self.drain_search() + search.evaluate(SEARCH_LINKS_JS, self.cfg.platform):
                    if topic_id_from_url(self.cfg.platform, t["url"]) == t["id"]: topics.setdefault(t["id"], t)
            if not topics: raise SafeStop("搜索页未读到主题；可能无结果、需登录或页面结构已变化，未生成伪数据")
            for topic in list(topics.values())[:self.cfg.max_topics]:
                if store.done(topic["id"]):
                    self.log(f"已完成该有限窗口，跳过主题 {topic['id']}"); continue
                self.collect_comments(detail, topic, query, store); store.export()
        finally:
            self.topic = None
            search.close(); detail.close()

    def run(self):
        self.cfg.validate()
        from playwright.sync_api import sync_playwright
        reports = []
        with sync_playwright() as p:
            opts = {"headless": False, "chromium_sandbox": True}
            if self.cfg.browser_channel: opts["channel"] = self.cfg.browser_channel
            browser = p.chromium.launch(**opts)
            context = browser.new_context(locale="zh-CN", timezone_id="Asia/Shanghai")
            # New ephemeral context. Browser credentials and storage_state are never saved.
            try:
                for query in dict.fromkeys(q.strip() for q in self.cfg.queries):
                    store = RecordStore(self.cfg, query)
                    status, reason = "window_finished", ""
                    try:
                        self.log(f"{self.cfg.platform}｜搜索词：{query}")
                        self.collect_query(context, query, store)
                        if store.incomplete_topics():
                            status, reason = "partial", "有主题没有捕获到可核验评论，不能解释为真正零评论；可重新运行未完成主题"
                    except SafeStop as e:
                        status, reason = "stopped", str(e)
                    except Exception as e:
                        # Only the error type is persisted; Playwright tracebacks can contain signed URLs.
                        status, reason = "error", f"{type(e).__name__}：浏览器操作失败。请检查页面、网络和浏览器安装；没有伪造采集结果"
                    finally:
                        result = store.export()
                        if result["export_errors"]:
                            status, reason = "export_error", "部分导出失败（可能文件正被Excel占用）；原文已存SQLite检查点。关闭占用后重跑。" + ", ".join(result["export_errors"])
                        result.update(status=status, reason=reason)
                        reports.append(result); store.close()
                    if status != "window_finished": break
            finally:
                try: context.close()
                except Exception: pass
                try: browser.close()
                except Exception: pass
        return reports


_RUN_LOCK = threading.Lock()


def run_collection(config, input_fn=input):
    """Notebook-safe sync wrapper; all Playwright operations stay on one owner thread."""
    config.validate()
    if not _RUN_LOCK.acquire(blocking=False):
        reason = "上一次采集仍在关闭浏览器或保存检查点，请等它结束再运行"
        print(reason)
        return [{"status":"blocked", "reason":reason}]
    messages, replies = queue.Queue(), queue.Queue()
    cancelled = threading.Event()
    def worker():
        try:
            result = BrowserCollector(config, messages, replies, cancelled).run()
            messages.put(("result", result))
        except Exception as e:
            # Do not return a verbose browser log that could expose session URLs.
            messages.put(("failure", f"{type(e).__name__}：无法启动浏览器或初始化采集。请检查playwright、Chromium安装及本机桌面环境；已有SQLite检查点不受影响。"))
        finally:
            _RUN_LOCK.release()
    thread = threading.Thread(target=worker, name="page-collector", daemon=True)
    thread.start()
    try:
        while True:
            kind, data = messages.get()
            if kind == "progress": print(data)
            elif kind == "input": replies.put(input_fn(data))
            elif kind == "failure":
                print(data); return [{"status":"blocked", "reason":data}]
            else:
                for report in data:
                    print(f"{report['query']}：{report['status']}，累计 {report['records']} 条")
                    if report["reason"]: print(report["reason"])
                    print("CSV：", report["csv"])
                return data
    except (KeyboardInterrupt, EOFError):
        cancelled.set(); replies.put("q")
        print("已请求停止。已提交的SQLite检查点可恢复；下次运行会重新导出CSV。")
        thread.join(timeout=5)
        return [{"status":"cancel_requested", "reason":"用户中止；请等待浏览器关闭再重跑"}]


def dependency_check():
    """Read-only check. Does not install software or launch a browser."""
    from importlib.util import find_spec
    from importlib.metadata import version
    if find_spec("playwright") is None:
        return {"playwright_installed":False, "next_step":"在当前Notebook内核安装：%pip install playwright tzdata，然后运行 python -m playwright install chromium"}
    return {"playwright_installed":True, "version":version("playwright"), "next_step":"首次使用需安装Chromium：python -m playwright install chromium；或BROWSER_CHANNEL='chrome'使用已安装Chrome"}
