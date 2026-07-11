#!/usr/bin/env python3
"""
SentryLive demo launcher.

Usage:
    python run_demo.py --google-api-key YOUR_KEY --pageindex-api-key YOUR_KEY

API keys can also come from existing GOOGLE_API_KEY / PAGEINDEX_API_KEY
environment variables, or you'll be prompted for them interactively if
neither is supplied.

Guidelines are pre-indexed (data/guidelines/doc_ids.json is bundled), so the
server starts immediately -- your PAGEINDEX_API_KEY must belong to the account
that indexed them. Delete doc_ids.json first if you'd rather index the bundled
PDFs under your own PageIndex account instead (one-time, a few minutes, uses
your own API credits).
"""
import argparse
import getpass
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DOC_IDS_PATH = BASE_DIR / "data" / "guidelines" / "doc_ids.json"

_banner_shown = False


def parse_args():
    p = argparse.ArgumentParser(description="Launch the SentryLive demo.")
    p.add_argument("--google-api-key", default=None,
                    help="Google Gemini API key (or set GOOGLE_API_KEY)")
    p.add_argument("--pageindex-api-key", default=None,
                    help="PageIndex API key (or set PAGEINDEX_API_KEY)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--no-browser", action="store_true",
                    help="Do not auto-open a browser tab")
    p.add_argument("--skip-index", action="store_true",
                    help="Skip guideline indexing even if doc_ids.json is missing "
                         "(the demo will run with no retrievable guidelines)")
    return p.parse_args()


def resolve_key(cli_value, env_name, prompt_label):
    if cli_value:
        return cli_value
    if os.getenv(env_name):
        return os.getenv(env_name)

    # Neither a --flag nor an environment variable was given -- prompt for it
    # directly rather than falling back to any placeholder value. Uses getpass
    # (not input()) so the key is never echoed to the terminal -- important if
    # you're recording this screen for a demo.
    global _banner_shown
    if not _banner_shown:
        print()
        print("No API keys were supplied via --google-api-key / --pageindex-api-key")
        print("or the GOOGLE_API_KEY / PAGEINDEX_API_KEY environment variables.")
        print("Enter them now -- input is hidden and nothing is stored anywhere")
        print(f"except the local .env file this script writes ({ENV_PATH.name}).")
        print()
        _banner_shown = True

    try:
        return getpass.getpass(f"  {prompt_label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def write_env_file(google_key, pageindex_key):
    lines = [
        f"GOOGLE_API_KEY={google_key}",
        f"PAGEINDEX_API_KEY={pageindex_key}",
        "CONCEPT_EXTRACTION_MODEL=gemini-2.5-flash",
        "ANSWER_GENERATION_MODEL=gemini-2.5-flash",
        "RETRIEVAL_MODEL=gemini-2.5-flash",
        "TEMPERATURE=0.1",
    ]
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {ENV_PATH} -- contains your API keys, keep this local "
          f"(do not commit or share it)")


def main():
    args = parse_args()

    google_key = resolve_key(args.google_api_key, "GOOGLE_API_KEY", "Google Gemini API key")
    pageindex_key = resolve_key(args.pageindex_api_key, "PAGEINDEX_API_KEY", "PageIndex API key")

    if not google_key or not pageindex_key:
        print("Both a Google Gemini API key and a PageIndex API key are required. Exiting.")
        sys.exit(1)

    write_env_file(google_key, pageindex_key)
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["PAGEINDEX_API_KEY"] = pageindex_key

    if not DOC_IDS_PATH.exists() and not args.skip_index:
        print()
        print("First-time setup: indexing the bundled guideline PDFs under your PageIndex account.")
        print("This uploads ~56 PDFs, runs once, takes a few minutes, and uses PageIndex API credits.")
        print()
        result = subprocess.run([sys.executable, "upload_guidelines.py"], cwd=BASE_DIR)
        if result.returncode != 0:
            print("Guideline indexing failed -- see the output above, then re-run this script.")
            sys.exit(1)
    elif DOC_IDS_PATH.exists():
        print(f"Found existing {DOC_IDS_PATH.name} -- skipping guideline indexing.")
        print("NOTE: this package ships pre-indexed guidelines. Your PAGEINDEX_API_KEY must belong")
        print("to the PageIndex account that indexed them, or retrieval will fail. If you want to")
        print("index under your own PageIndex account instead, delete data/guidelines/doc_ids.json")
        print("and re-run this script.")

    print(f"\nStarting the SentryLive server at http://{args.host}:{args.port}\n")

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", args.host, "--port", str(args.port)],
        cwd=BASE_DIR,
    )

    if not args.no_browser:
        time.sleep(2.5)
        webbrowser.open(f"http://{args.host}:{args.port}")

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.terminate()


if __name__ == "__main__":
    main()
