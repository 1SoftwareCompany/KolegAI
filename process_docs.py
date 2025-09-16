#!/usr/bin/env python3
# clean_notion_export.py
# Purpose: Filter a Notion export for RAG ingestion:
# - Drop trivial/short pages
# - Remove Jira tickets
# - Drop glossary/field-spec style stubs
# - Deduplicate near-identical docs
# - Chunk by headings
# - Emit JSONL with metadata
# - Optionally move/delete dropped files (trash dir or hard delete)

import argparse
import hashlib
import json
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

MD_EXTS = {".md", ".markdown"}
HTML_EXTS = {".html", ".htm"}

ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b")  # e.g., ABC-12345
JIRA_FIELD_HINTS = {
    "Issue key", "Key", "Issue Type", "Type", "Reporter", "Assignee",
    "Priority", "Status", "Resolution", "Affects Version", "Fix Version",
    "Sprint", "Story Points", "Epic Link", "Components", "Labels"
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^```")
LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

TOC_HINTS = {"table of contents", "toc", "index"}


# ---------- Utils ----------

def approx_token_count(text: str) -> int:
    # Simple word-ish estimate
    return len(text.split())


def strip_markdown(text: str) -> str:
    t = LINK_RE.sub(r"\1", text)  # keep link text, drop URLs
    t = re.sub(r"`{1,3}[^`]+`{1,3}", "", t)  # inline code
    t = re.sub(r"^>.*$", "", t, flags=re.MULTILINE)  # blockquotes
    t = re.sub(r"^[-*]\s+", "", t, flags=re.MULTILINE)  # bullets
    t = re.sub(r"^\d+\.\s+", "", t, flags=re.MULTILINE)  # numbered lists
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)  # headings
    t = HTML_TAG_RE.sub("", t)  # strip html tags
    return t


def normalize_for_hash(text: str) -> str:
    # Normalize aggressively for dedup: lowercase, strip punctuation/whitespace
    t = text.lower()
    t = HTML_TAG_RE.sub(" ", t)
    t = re.sub(r"[^\w]+", "", t)  # keep letters/digits/underscore
    return t


def sha1_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def guess_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            return m.group(2).strip()
    return Path(fallback).stem.replace("_", " ").strip()


def is_probably_toc(text: str) -> bool:
    stripped = strip_markdown(text).lower()
    if sum(ch.isalnum() for ch in stripped) < 30:
        return True
    for hint in TOC_HINTS:
        if hint in stripped and approx_token_count(stripped) < 50:
            return True
    return False


def has_code_block(lines: List[str]) -> bool:
    fence_count = 0
    for ln in lines:
        if CODE_FENCE_RE.match(ln.strip()):
            fence_count += 1
            if fence_count >= 2:  # opening + closing
                return True
    return False


def contains_must_keep(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    for kw in keywords:
        kw = kw.strip().lower()
        if kw and kw in t:
            return True
    return False


def is_jira_ticket(text: str, jira_host: str = None) -> bool:
    score = 0
    if ISSUE_KEY_RE.search(text):
        score += 1
    if jira_host and jira_host.lower() in text.lower():
        score += 1
    if "atlassian.net/browse/" in text.lower():
        score += 1
    # Look for common field labels near the top
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    field_hits = 0
    for ln in lines[:80]:
        if ":" in ln or "|" in ln or "-" in ln:
            for fh in JIRA_FIELD_HINTS:
                if re.search(rf"\b{re.escape(fh)}\b", ln, re.IGNORECASE):
                    field_hits += 1
                    if field_hits >= 2:
                        break
        if field_hits >= 2:
            break
    if field_hits >= 2:
        score += 1
    # Old Jira wiki markup like "h3. Summary"
    if re.search(r"\bh\d\.\s", text):
        score += 1
    return score >= 2  # require at least two signals


def chunk_by_headings(lines: List[str], max_chars: int = 2500) -> List[Tuple[str, str]]:
    """
    Returns list of (section_title, section_text).
    Uses headings as hard boundaries; also splits oversized sections.
    """
    sections: List[Tuple[str, str]] = []
    cur_title = None
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_title, cur_buf, sections
        if not cur_buf:
            return
        text = "\n".join(cur_buf).strip()
        if not text:
            cur_buf = []
            return
        if len(text) > max_chars:
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            accum = ""
            for p in parts:
                if len(accum) + len(p) + 2 <= max_chars:
                    accum = (accum + "\n\n" + p) if accum else p
                else:
                    sections.append((cur_title or "", accum))
                    accum = p
            if accum:
                sections.append((cur_title or "", accum))
        else:
            sections.append((cur_title or "", text))
        cur_buf = []

    for ln in lines:
        m = HEADING_RE.match(ln.strip())
        if m:
            flush()
            cur_title = m.group(2).strip()
            cur_buf.append(ln)  # keep the heading in the section for context
        else:
            cur_buf.append(ln)
    flush()
    return sections or [("", "\n".join(lines).strip())]


def read_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def should_drop_short(raw_text: str, min_words: int, min_chars: int) -> bool:
    stripped = strip_markdown(raw_text)
    if approx_token_count(stripped) < min_words:
        return True
    if len(stripped) < min_chars:
        return True
    return False


def collect_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (MD_EXTS | HTML_EXTS):
            yield p


# ---------- Glossary/Field-spec stub detection ----------

def count_headings(text: str) -> int:
    return sum(1 for ln in text.splitlines() if HEADING_RE.match(ln.strip()))


def _kv_line(ln: str) -> bool:
    # key: value with relatively short key
    return bool(re.match(r"^[A-Za-z][\w\s/()-]{0,40}:\s+\S", ln))


def is_glossary_stub(text: str, title: str, max_words: int, deny_title_re: str) -> bool:
    stripped = strip_markdown(text)
    if approx_token_count(stripped) > max_words:
        return False

    hcount = count_headings(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    colon_lines = sum(1 for ln in lines if _kv_line(ln))
    ratio = colon_lines / max(1, len(lines))

    signals = 0
    if hcount <= 2:
        signals += 1
    if ratio >= 0.4:
        signals += 1
    if re.search(r"\b(Mandatory|Required|Description|Default|Type|Format|Example|Allowed Values)\b",
                 stripped, re.I):
        signals += 1
    if deny_title_re and re.search(deny_title_re, (title or "").strip()):
        signals += 1
    first_para = re.split(r"\n\s*\n", stripped, maxsplit=1)[0]
    if approx_token_count(first_para) <= 35:
        signals += 1

    return signals >= 3


# ---------- File drop/move helpers ----------

def drop_file(fp: Path, rel_path: str, args):
    if args.dry_run:
        return
    if args.trash_dir:
        dest = Path(args.trash_dir).expanduser().resolve() / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fp), str(dest))
    elif args.hard_delete:
        try:
            fp.unlink()
        except Exception:
            pass
    # else: leave file in place


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Clean Notion export for RAG ingestion.")
    ap.add_argument("--input", required=True, help="Path to Notion export root")
    ap.add_argument("--output", required=True, help="Output JSONL file (chunks)")
    ap.add_argument("--min-words", type=int, default=40, help="Min words to keep a page")
    ap.add_argument("--min-chars", type=int, default=200, help="Min characters to keep a page")
    ap.add_argument("--max-chunk-chars", type=int, default=2500, help="Max characters per chunk")
    ap.add_argument("--drop-jira", action="store_true", help="Drop pages detected as Jira tickets")
    ap.add_argument("--jira-host", type=str, default=None, help="e.g., yourcompany.atlassian.net")
    ap.add_argument("--keep-short-if-code", action="store_true",
                    help="Keep short pages if they contain code fences")
    ap.add_argument("--must-keep", type=str, default="",
                    help="Comma-separated keywords that can rescue short pages")
    ap.add_argument("--drop-glossary-stubs", action="store_true",
                    help="Drop short dictionary/field-spec style pages")
    ap.add_argument("--glossary-max-words", type=int, default=150,
                    help="Max words to still treat as a glossary stub")
    ap.add_argument("--deny-title-regex", type=str,
                    default=r"^\{?[A-Z0-9_]{3,}\}?$",
                    help="Regex for field-like titles to drop (e.g., {CATALOG_TYPE})")
    ap.add_argument("--trash-dir", type=str, default=None,
                    help="Move dropped files here (safe & reversible)")
    ap.add_argument("--hard-delete", action="store_true",
                    help="Permanently delete dropped files (careful)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Analyze only; do not write output or move/delete files")
    args = ap.parse_args()

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        print(f"Input path not found: {root}", file=sys.stderr)
        sys.exit(1)

    must_keep_keywords = [s.strip() for s in args.must_keep.split(",") if s.strip()]

    files = list(collect_files(root))
    if not files:
        print("No Markdown/HTML files found.")
        sys.exit(0)

    seen_hashes = set()
    stats = {
        "total_files": 0,
        "kept_docs": 0,
        "emitted_chunks": 0,
        "dropped_short": 0,
        "dropped_jira": 0,
        "dropped_toc": 0,
        "dropped_glossary_stub": 0,
        "dropped_duplicate": 0,
        "empty_or_unreadable": 0,
    }

    out_path = Path(args.output).expanduser().resolve()
    out_f = None if args.dry_run else out_path.open("w", encoding="utf-8")

    for fp in files:
        stats["total_files"] += 1
        raw = read_text(fp)
        rel_path = str(fp.relative_to(root))

        if not raw or not raw.strip():
            stats["empty_or_unreadable"] += 1
            drop_file(fp, rel_path, args)
            continue

        title = guess_title(raw, fallback=fp.name)

        # Quick thin/TOC page detection
        if is_probably_toc(raw):
            stats["dropped_toc"] += 1
            drop_file(fp, rel_path, args)
            continue

        # Short-page logic with rescue conditions
        drop_for_short = should_drop_short(raw, args.min_words, args.min_chars)
        lines = raw.splitlines()

        rescued = False
        if drop_for_short and args.keep_short_if_code and has_code_block(lines):
            rescued = True
        if drop_for_short and contains_must_keep(raw, must_keep_keywords):
            rescued = True
        if drop_for_short and not rescued:
            stats["dropped_short"] += 1
            drop_file(fp, rel_path, args)
            continue

        # Glossary/field-spec stub detection
        if args.drop_glossary_stubs and is_glossary_stub(
            raw, title, args.glossary_max_words, args.deny_title_regex
        ):
            stats["dropped_glossary_stub"] += 1
            drop_file(fp, rel_path, args)
            continue

        # Jira detection
        if args.drop_jira and is_jira_ticket(raw, jira_host=args.jira_host):
            stats["dropped_jira"] += 1
            drop_file(fp, rel_path, args)
            continue

        # Dedup (document-level)
        normalized = normalize_for_hash(raw)
        h = sha1_hash(normalized)
        if h in seen_hashes:
            stats["dropped_duplicate"] += 1
            drop_file(fp, rel_path, args)
            continue
        seen_hashes.add(h)

        # Chunk and emit JSONL
        sections = chunk_by_headings(lines, max_chars=args.max_chunk_chars)
        doc_id = h[:16]
        aka_title = title or Path(rel_path).stem

        emitted_here = 0
        for idx, (sec_title, sec_text) in enumerate(sections):
            stripped = strip_markdown(sec_text)
            if approx_token_count(stripped) == 0 or len(stripped) < 10:
                continue  # skip empty-ish chunks
            rec = {
                "doc_id": doc_id,
                "section_id": idx,
                "title": aka_title,
                "section_title": sec_title,
                "path": rel_path,
                "ext": fp.suffix.lower(),
                "word_count": approx_token_count(stripped),
                "char_count": len(stripped),
                "text": sec_text.strip(),
                "source": "notion-export",
            }
            if out_f:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            emitted_here += 1

        if emitted_here > 0:
            stats["kept_docs"] += 1
            stats["emitted_chunks"] += emitted_here
        else:
            # Nothing worth emitting; treat as short/empty content
            stats["dropped_short"] += 1
            drop_file(fp, rel_path, args)

    if out_f:
        out_f.close()

    # Report
    print("\n=== Clean Notion Export Report ===")
    print(f"Scanned files         : {stats['total_files']}")
    print(f"Kept documents        : {stats['kept_docs']}")
    print(f"Emitted chunks        : {stats['emitted_chunks']}")
    print(f"Dropped (short)       : {stats['dropped_short']}")
    print(f"Dropped (glossary)    : {stats['dropped_glossary_stub']}")
    print(f"Dropped (Jira)        : {stats['dropped_jira']}")
    print(f"Dropped (TOC/index)   : {stats['dropped_toc']}")
    print(f"Dropped (duplicates)  : {stats['dropped_duplicate']}")
    print(f"Dropped (empty/unread): {stats['empty_or_unreadable']}")
    if not args.dry_run:
        print(f"\nOutput JSONL: {out_path}")
    else:
        print("\n(Dry run: no output written and no files moved/deleted)")


if __name__ == "__main__":
    main()
