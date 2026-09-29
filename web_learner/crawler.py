from __future__ import annotations

import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from urllib.request import HTTPRedirectHandler, Request, build_opener


class TextExtractor(HTMLParser):
    """Small dependency-free HTML to readable text extractor."""
    SKIP = {"script", "style", "noscript", "svg", "canvas", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self.skip:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title:
            self.title += (" " if self.title else "") + text
        self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


class AllowlistedRedirectHandler(HTTPRedirectHandler):
    """Reject a redirect before the client requests an unapproved host."""

    def __init__(self, allowed):
        super().__init__()
        self.allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.allowed(newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicWebLearner:
    """Fetch allowlisted public pages and turn them into training records.

    This intentionally does not execute downloaded code, install packages, or
    treat arbitrary pages as trusted instructions. Sources are configured by
    the owner, and every record keeps provenance for later auditing/deletion.
    """

    def __init__(self, config_path: str = "config/web_learning.yaml"):
        import yaml
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}
        self.knowledge_dir = Path(self.cfg.get("storage", {}).get("knowledge_dir", "data/web_knowledge"))
        self.raw_dir = Path(self.cfg.get("storage", {}).get("raw_dir", "data/web_raw"))
        self.manifest_path = Path(self.cfg.get("storage", {}).get("manifest", "data/web_knowledge/manifest.jsonl"))
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.seen = self._load_seen()
        self.robots_cache: dict[str, RobotFileParser] = {}
        self.opener = build_opener(AllowlistedRedirectHandler(self._allowed))

    def _load_seen(self):
        seen = set()
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                try:
                    seen.add(json.loads(line)["content_hash"])
                except Exception:
                    pass
        return seen

    def _allowed(self, url: str) -> bool:
        p = urlparse(url)
        if p.scheme not in {"http", "https"}:
            return False
        domains = self.cfg.get("allowed_domains", [])
        if domains and not any(p.hostname == d or (p.hostname or "").endswith("." + d) for d in domains):
            return False
        return True

    def _robots(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        rp = self.robots_cache.get(base)
        if rp is None:
            rp = RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                # Network failure is fail-closed for crawling.
                return False
            self.robots_cache[base] = rp
        return rp.can_fetch(self.cfg.get("user_agent", "RaksamWebLearner/1.0"), url)

    def _save_raw_html(self, url: str, html: str) -> None:
        """Archive raw public input separately; raw records are never trained on."""
        if not self.cfg.get("save_raw_html", True):
            return
        record = {
            "source_url": url,
            "retrieved_at": time.time(),
            "content_hash": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "html": html,
        }
        out = self.raw_dir / f"raw_{time.strftime('%Y-%m-%d')}.jsonl"
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fetch(self, url: str) -> tuple[str, list[str], str, str] | None:
        if not self._allowed(url) or not self._robots(url):
            return None
        headers = {"User-Agent": self.cfg.get("user_agent", "RaksamWebLearner/1.0")}
        req = Request(url, headers=headers)
        try:
            with self.opener.open(req, timeout=float(self.cfg.get("request_timeout", 15))) as r:
                final_url = r.geturl()
                # Keep the final URL in the record and fail closed when either
                # the allowlist or that site's robots policy rejects it.
                if not self._allowed(final_url) or not self._robots(final_url):
                    return None
                ctype = r.headers.get_content_type()
                if ctype != "text/html":
                    return None
                max_bytes = int(self.cfg.get("max_page_bytes", 2_000_000))
                raw = r.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    return None
                html = raw.decode(r.headers.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            return None
        self._save_raw_html(final_url, html)
        parser = TextExtractor()
        parser.feed(html)
        text = re.sub(r"\n{3,}", "\n\n", parser.text()).strip()
        links = []
        for href in parser.links:
            child = urldefrag(urljoin(final_url, href))[0]
            if self._allowed(child):
                links.append(child)
        return text, links, parser.title.strip(), final_url

    def _quality_ok(self, text: str) -> bool:
        min_chars = int(self.cfg.get("min_text_chars", 800))
        if len(text) < min_chars:
            return False
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if len(lines) < 5:
            return False
        alpha = sum(c.isalpha() for c in text)
        return alpha / max(1, len(text)) >= 0.35

    def ingest(self, url: str, text: str, title: str = "") -> bool:
        if not self._quality_ok(text):
            return False
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in self.seen:
            return False
        record = {
            "source_url": url,
            "title": title,
            "content_hash": content_hash,
            "retrieved_at": time.time(),
            "text": text,
        }
        day = time.strftime("%Y-%m-%d")
        out = self.knowledge_dir / f"web_{day}.jsonl"
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: record[k] for k in ("source_url", "title", "content_hash", "retrieved_at")}, ensure_ascii=False) + "\n")
        self.seen.add(content_hash)
        return True

    def crawl(self) -> dict:
        sources = self.cfg.get("sources", [])
        max_pages = int(self.cfg.get("max_pages_per_run", 20))
        max_depth = int(self.cfg.get("max_depth", 1))
        delay = float(self.cfg.get("delay_seconds", 1.0))
        queue = [(s, 0) for s in sources]
        visited = set()
        fetched = accepted = 0
        while queue and fetched < max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            result = self.fetch(url)
            if result is None:
                continue
            fetched += 1
            text, links, title, final_url = result
            accepted += int(self.ingest(final_url, text, title))
            if depth < max_depth:
                for link in links:
                    if link not in visited and len(queue) < max_pages * 4:
                        queue.append((link, depth + 1))
            time.sleep(delay)
        return {"fetched": fetched, "accepted": accepted, "visited": len(visited)}
