#!/usr/bin/env python3
# clean_notion_export_soft.py
# Purpose: Softer filter for a Notion export for RAG ingestion.
# - Keeps far more docs while removing the worst junk
# - Tunable profiles: gentle (default) / balanced / aggressive
# - Always drops files with "release notes"/"changelog" in the NAME
# - Optionally drops content that *looks like* release notes (unless it clearly looks like services/APIs)
# - Strong rescue rules for services/APIs/architecture/integration docs
# - Jira/TOC/glossary/short-page detection is less harsh and easily overridden
# - Dedup exact or normalized text (configurable)
# - Chunk by headings
# - Emits JSONL with rich metadata + optional JSONL sample of dropped docs with reasons
#
# Examples:
#   python3 clean_notion_export_soft.py \
#       --input ./Notion-Export \
#       --output ./cleaned.jsonl \
#       --profile gentle \
#       --drop-jira \
#       --drop-glossary-stubs \
#       --dump-dropped ./dropped_sample.jsonl \
#       --dump-dropped-limit 500
#
#   python3 clean_notion_export_soft.py --input ./Notion-Export --output ./cleaned.jsonl --profile balanced
#

import argparse
import hashlib
import json
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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

# --- Release Notes/changelog detection ---
FILENAME_DENY_RE = re.compile(
    r"(?i)(^|[/\\])[^/\\]*\b(release[ _-]?notes|changelog|change[ _-]?log)\b[^/\\]*\.(md|markdown|html?|rst)$"
)

def looks_like_release_notes(text: str) -> bool:
    t = text.lower()
    signals = 0
    if "release notes" in t: signals += 1
    if "release number" in t: signals += 1
    if "release date" in t: signals += 1
    if "customers might notice" in t or "admins might notice" in t: signals += 1
    if "changes to core" in t or "what changed" in t: signals += 1
    if re.search(r"\b(version|v\d+\.\d+)\b", t): signals += 1
    return signals >= 2

# Strong rescue if it looks like services/APIs/architecture/integration
RESCUE_HINTS = [
    "microservice", "microservices", "service", "services", "api", "apis",
    "architecture", "integration", "sequence diagram", "participant ",
    "cronus", "iport", "ieventhandler", "publisher", "subscriber",
    "opc api", "opc web", "newgen", "iaa", "gateway", "adapter", "ingestion",
]

def strong_rescue_signal(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in RESCUE_HINTS)


# ---------- Utils ----------

def approx_token_count(text: str) -> int:
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
    t = text.lower()
    t = HTML_TAG_RE.sub(" ", t)
    t = re.sub(r"[^\w]+", "", t)
    return t


def sha1_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def guess_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            return m.group(2).strip()
    return Path(fallback).stem.replace("_", " ").strip()


def is_probably_toc(text: str, profile: str = "gentle") -> bool:
    # Softer heuristic: only drop if explicit hints AND very short
    stripped = strip_markdown(text).lower()
    alnum = sum(ch.isalnum() for ch in stripped)
    if profile == "aggressive":
        short_ok = approx_token_count(stripped) < 80 or alnum < 60
    elif profile == "balanced":
        short_ok = approx_token_count(stripped) < 60 or alnum < 50
    else:  # gentle
        short_ok = approx_token_count(stripped) < 45 and alnum < 40
    if short_ok:
        for hint in TOC_HINTS:
            if hint in stripped:
                return True
    return False


def has_code_block(lines: List[str]) -> bool:
    fence_count = 0
    for ln in lines:
        if CODE_FENCE_RE.match(ln.strip()):
            fence_count += 1
            if fence_count >= 2:
                return True
    return False


def contains_must_keep(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    for kw in keywords:
        kw = kw.strip().lower()
        if kw and kw in t:
            return True
    return False


def is_jira_ticket(text: str, jira_host: Optional[str] = None, profile: str = "gentle") -> bool:
    # Softer scoring: require at least 2 signals in gentle, 3 in aggressive
    score = 0
    t = text.lower()
    if ISSUE_KEY_RE.search(text):
        score += 1
    if jira_host and jira_host.lower() in t:
        score += 1
    if "atlassian.net/browse/" in t:
        score += 1

    # Look for Jira-ish field lines near the top
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    field_hits = 0
    for ln in lines[:120]:
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

    # Hierarchical IDs like H1., H2., etc.
    if re.search(r"\bh\d\.\s", text):
        score += 1

    needed = 2 if profile == "gentle" else (3 if profile == "balanced" else 3)
    return score >= needed


def chunk_by_headings(lines: List[str], max_chars: int = 2500) -> List[Tuple[str, str]]:
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
            cur_buf.append(ln)
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
    return bool(re.match(r"^[A-Za-z][\w\s/()-]{0,60}:\s+\S", ln))


def is_glossary_stub(text: str, title: str, max_words: int, deny_title_re: Optional[str], signals_needed: int) -> bool:
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
    if ratio >= 0.35:
        signals += 1
    if re.search(r"\b(Mandatory|Required|Description|Default|Type|Format|Example|Allowed Values)\b",
                 stripped, re.I):
        signals += 1
    if deny_title_re and re.search(deny_title_re, (title or "").strip()):
        signals += 1
    first_para = re.split(r"\n\s*\n", stripped, maxsplit=1)[0]
    if approx_token_count(first_para) <= 35:
        signals += 1
    return signals >= signals_needed


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


def record_dropped(container: List[Dict], rel_path: str, reason: str, title: str, sample: str, limit: int):
    if container is None:
        return
    if len(container) >= limit:
        return
    container.append({
        "path": rel_path,
        "reason": reason,
        "title": title,
        "sample": sample[:400]
    })


# ---------- Profiles ----------

def profile_defaults(profile: str) -> Dict:
    profile = (profile or "gentle").lower()
    if profile == "aggressive":
        return dict(min_words=50, min_chars=240, glossary_max_words=140, glossary_signals=3)
    if profile == "balanced":
        return dict(min_words=35, min_chars=160, glossary_max_words=130, glossary_signals=3)
    # gentle
    return dict(min_words=25, min_chars=120, glossary_max_words=120, glossary_signals=4)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Soft-clean Notion export for RAG ingestion.")
    ap.add_argument("--input", required=True, help="Path to Notion export root")
    ap.add_argument("--output", required=True, help="Output JSONL file (chunks)")
    ap.add_argument("--profile", choices=["gentle", "balanced", "aggressive"], default="gentle")

    ap.add_argument("--min-words", type=int, default=None, help="Override min words to keep a page")
    ap.add_argument("--min-chars", type=int, default=None, help="Override min characters to keep a page")
    ap.add_argument("--max-chunk-chars", type=int, default=2500, help="Max characters per chunk")

    ap.add_argument("--drop-jira", action="store_true", help="Drop pages detected as Jira tickets")
    ap.add_argument("--jira-host", type=str, default=None, help="e.g., yourcompany.atlassian.net")

    ap.add_argument("--keep-short-if-code", action="store_true", help="Keep short pages if they contain code fences")
    ap.add_argument("--must-keep", type=str, default="",
                    help="Comma-separated keywords that can rescue short pages or release-notes-like content")

    ap.add_argument("--drop-glossary-stubs", action="store_true",
                    help="Drop short dictionary/field-spec style pages")
    ap.add_argument("--glossary-max-words", type=int, default=None,
                    help="Max words to still treat as a glossary stub (profile default if omitted)")
    ap.add_argument("--deny-title-regex", type=str,
                    default=r"^\{?[A-Z0-9_]{3,}\}?$",
                    help="Regex for field-like titles to drop")

    ap.add_argument("--drop-hawk", action="store_true", help="Drop any document mentioning HAWK")

    ap.add_argument("--dedup-mode", choices=["none", "exact", "normalized"], default="normalized",
                    help="Duplicate detection mode")

    ap.add_argument("--dump-dropped", type=str, default=None, help="Write a JSONL sample of dropped docs")
    ap.add_argument("--dump-dropped-limit", type=int, default=500, help="Max dropped samples to write")

    ap.add_argument("--trash-dir", type=str, default=None, help="Move dropped files here")
    ap.add_argument("--hard-delete", action="store_true", help="Permanently delete dropped files")
    ap.add_argument("--dry-run", action="store_true", help="Analyze only; no output or file ops")

    args = ap.parse_args()

    prof = profile_defaults(args.profile)
    min_words = args.min_words if args.min_words is not None else prof["min_words"]
    min_chars = args.min_chars if args.min_chars is not None else prof["min_chars"]
    glossary_max_words = args.glossary_max_words if args.glossary_max_words is not None else prof["glossary_max_words"]
    glossary_signals = prof["glossary_signals"]

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        print(f"Input path not found: {root}", file=sys.stderr)
        sys.exit(1)

    must_keep_keywords = [s.strip() for s in args.must_keep.split(",") if s.strip()]
    # Seed rescue keywords for services/APIs/architecture
    seed_rescue = [
        "microservice", "service", "api", "architecture", "integration",
        "sequence diagram", "participant ", "cronus", "opc", "newgen", "iaa",
        "gateway", "adapter", "ingestion"
    ]
    for k in seed_rescue:
        if k not in must_keep_keywords:
            must_keep_keywords.append(k)

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
        "dropped_toc_or_index": 0,
        "dropped_glossary_stub": 0,
        "dropped_duplicate": 0,
        "dropped_hawk": 0,
        "dropped_release_notes_name": 0,
        "dropped_release_notes_content": 0,
        "empty_or_unreadable": 0,
    }

    out_path = Path(args.output).expanduser().resolve()
    out_f = None if args.dry_run else out_path.open("w", encoding="utf-8")

    dropped_samples: Optional[List[Dict]] = [] if args.dump_dropped else None

    for fp in files:
        stats["total_files"] += 1
        raw = read_text(fp)
        rel_path = str(fp.relative_to(root))

        if not raw or not raw.strip():
            stats["empty_or_unreadable"] += 1
            record_dropped(dropped_samples, rel_path, "empty_or_unreadable", fp.name, "", args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        title = guess_title(raw, fallback=fp.name)

        # 1) Hard drop by filename if looks like release notes / changelog
        if FILENAME_DENY_RE.search(rel_path):
            stats["dropped_release_notes_name"] += 1
            record_dropped(dropped_samples, rel_path, "release_notes_name", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 2) Drop obvious TOC/Index (softer by profile)
        if is_probably_toc(raw, profile=args.profile):
            stats["dropped_toc_or_index"] += 1
            record_dropped(dropped_samples, rel_path, "toc_or_index", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 3) Short-page drop with rescues
        drop_for_short = should_drop_short(raw, min_words, min_chars)
        lines = raw.splitlines()
        rescued = False
        if drop_for_short and args.keep_short_if_code and has_code_block(lines):
            rescued = True
        if drop_for_short and contains_must_keep(raw, must_keep_keywords):
            rescued = True
        if drop_for_short and not rescued:
            stats["dropped_short"] += 1
            record_dropped(dropped_samples, rel_path, "short_page", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 4) Optional HAWK drop
        if args.drop_hawk and re.search(r"\bhawk\b", raw, re.IGNORECASE):
            stats["dropped_hawk"] += 1
            record_dropped(dropped_samples, rel_path, "hawk", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 5) Optional glossary stub drop (stricter in gentle profile: needs more signals)
        if args.drop_glossary_stubs and is_glossary_stub(
            raw, title, glossary_max_words, args.deny_title_regex, signals_needed=glossary_signals
        ):
            stats["dropped_glossary_stub"] += 1
            record_dropped(dropped_samples, rel_path, "glossary_stub", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 6) Optional Jira drop (soft)
        if args.drop_jira and is_jira_ticket(raw, jira_host=args.jira_host, profile=args.profile):
            stats["dropped_jira"] += 1
            record_dropped(dropped_samples, rel_path, "jira_like", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 7) Drop by content if it *reads* like release notes, unless we have strong rescue signal
        if looks_like_release_notes(raw) and not strong_rescue_signal(raw):
            stats["dropped_release_notes_content"] += 1
            record_dropped(dropped_samples, rel_path, "release_notes_content", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)
            continue

        # 8) Dedup
        if args.dedup_mode != "none":
            if args.dedup_mode == "exact":
                hbase = raw
            else:
                hbase = normalize_for_hash(raw)
            h = sha1_hash(hbase)
            if h in seen_hashes:
                stats["dropped_duplicate"] += 1
                record_dropped(dropped_samples, rel_path, "duplicate", title, raw, args.dump_dropped_limit)
                drop_file(fp, rel_path, args)
                continue
            seen_hashes.add(h)
        else:
            h = sha1_hash(normalize_for_hash(raw))  # still use for doc_id stability

        # 9) Chunk and emit
        sections = chunk_by_headings(lines, max_chars=args.max_chunk_chars)
        doc_id = h[:16]
        aka_title = title or Path(rel_path).stem

        emitted_here = 0
        for idx, (sec_title, sec_text) in enumerate(sections):
            stripped = strip_markdown(sec_text)
            if approx_token_count(stripped) == 0 or len(stripped) < 10:
                continue
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
                "profile": args.profile,
            }
            if out_f:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            emitted_here += 1

        if emitted_here > 0:
            stats["kept_docs"] += 1
            stats["emitted_chunks"] += emitted_here
        else:
            stats["dropped_short"] += 1
            record_dropped(dropped_samples, rel_path, "empty_after_chunking", title, raw, args.dump_dropped_limit)
            drop_file(fp, rel_path, args)

    if out_f:
        out_f.close()

    # Dump dropped sample if requested
    if args.dump_dropped and dropped_samples is not None:
        dump_path = Path(args.dump_dropped).expanduser().resolve()
        with dump_path.open("w", encoding="utf-8") as df:
            for item in dropped_samples:
                df.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== Soft Clean Notion Export Report ===")
    print(f"Scanned files                 : {stats['total_files']}")
    print(f"Kept documents                : {stats['kept_docs']}")
    print(f"Emitted chunks                : {stats['emitted_chunks']}")
    print(f"Dropped (short)               : {stats['dropped_short']}")
    print(f"Dropped (glossary)            : {stats['dropped_glossary_stub']}")
    print(f"Dropped (Jira-like)           : {stats['dropped_jira']}")
    print(f"Dropped (TOC/index)           : {stats['dropped_toc_or_index']}")
    print(f"Dropped (duplicates)          : {stats['dropped_duplicate']}")
    print(f"Dropped (HAWK)                : {stats['dropped_hawk']}")
    print(f"Dropped (release notes name)  : {stats['dropped_release_notes_name']}")
    print(f"Dropped (release notes text)  : {stats['dropped_release_notes_content']}")
    print(f"Dropped (empty/unreadable)    : {stats['empty_or_unreadable']}")

    if not args.dry_run:
        print(f"\nOutput JSONL: {out_path}")
        if args.dump_dropped:
            print(f"Dropped sample JSONL: {Path(args.dump_dropped).expanduser().resolve()}")
    else:
        print("\n(Dry run: no output written and no files moved/deleted)")


if __name__ == "__main__":
    main()
