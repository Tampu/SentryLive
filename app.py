import os
import json
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SentryLive")

sessions: dict = {}

BASE_DIR          = Path(__file__).parent
GUIDELINES_DIR    = (BASE_DIR / "data" / "guidelines").resolve()
CONVERSATIONS_DIR = BASE_DIR / "data" / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR        = BASE_DIR / "static"


# ── Conversation helpers ────────────────────────────────────────────────────

def _conv_path(conv_id: str) -> Path:
    return CONVERSATIONS_DIR / f"{conv_id}.json"


def _load_conv(conv_id: str) -> dict | None:
    p = _conv_path(conv_id)
    return json.loads(p.read_text()) if p.exists() else None


def _save_conv(conv: dict) -> None:
    _conv_path(conv["id"]).write_text(json.dumps(conv, indent=2))


def _list_convs() -> list:
    result = []
    for p in sorted(CONVERSATIONS_DIR.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            result.append({
                "id":            data["id"],
                "title":         data.get("title", "Untitled"),
                "role":          data.get("role", "patient"),
                "updated_at":    data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            pass
    return result


# ── Models ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    role: str = "patient"
    force_refresh: bool = False


class FeedbackRequest(BaseModel):
    query: str
    original_answer: str
    correction: str


class CacheSaveRequest(BaseModel):
    query: str
    answer: str
    role: str = "patient"
    evidence: list = []
    verification: Optional[dict] = None
    temporal_verification: Optional[dict] = None


# ── Chat endpoint ────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    from pipeline.orchestrator import run_pipeline

    session_id = request.session_id or str(uuid.uuid4())

    # Load or create conversation
    if session_id not in sessions:
        conv = _load_conv(session_id)
        if conv:
            history = [
                {"query": m["query"], "answer": m["answer"]}
                for m in conv.get("messages", [])
            ]
            sessions[session_id] = {"history": history, "conv": conv}
        else:
            new_conv = {
                "id":           session_id,
                "title":        "",
                "role":         request.role,
                "created_at":   datetime.now(timezone.utc).isoformat(),
                "updated_at":   datetime.now(timezone.utc).isoformat(),
                "messages":     [],
                "all_evidence": [],
            }
            sessions[session_id] = {"history": [], "conv": new_conv}

    session = sessions[session_id]

    try:
        result = await run_in_threadpool(
            run_pipeline,
            query=request.query,
            role=request.role,
            conversation_history=session["history"],
            force_refresh=request.force_refresh,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n{'='*60}\nPIPELINE ERROR\n{'='*60}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "traceback": tb},
        )

    session["history"].append({"query": request.query, "answer": result["answer"]})

    # Persist to disk
    conv = session["conv"]
    if not conv["title"]:
        t = request.query
        conv["title"] = t[:60] + ("…" if len(t) > 60 else "")

    now = datetime.now(timezone.utc).isoformat()
    conv["messages"].append({
        "query":                 request.query,
        "answer":                result["answer"],
        "evidence":              result.get("evidence", []),
        "verification":          result.get("verification"),
        "temporal_verification": result.get("temporal_verification"),
        "cache_hit":             result.get("cache_hit", False),
        "timestamp":             now,
    })
    conv["updated_at"] = now

    # Accumulate evidence (de-dupe by filename+page)
    existing_keys = {(e.get("filename",""), e.get("page",0)) for e in conv["all_evidence"]}
    for ev in result.get("evidence", []):
        key = (ev.get("filename",""), ev.get("page", 0))
        if key not in existing_keys:
            conv["all_evidence"].append(ev)
            existing_keys.add(key)

    _save_conv(conv)

    return {
        "session_id":            session_id,
        "answer":                result["answer"],
        "evidence":              result.get("evidence", []),
        "all_evidence":          conv["all_evidence"],
        "verification":          result.get("verification"),
        "temporal_verification": result.get("temporal_verification"),
        "cache_hit":             result.get("cache_hit", False),
        "cached_query":          result.get("cached_query"),
    }


# ── Conversation CRUD ────────────────────────────────────────────────────────

@app.get("/api/conversations")
async def get_conversations():
    return {"conversations": _list_convs()}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = _load_conv(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    p = _conv_path(conv_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    p.unlink()
    sessions.pop(conv_id, None)
    return {"status": "deleted"}


# ── Other endpoints ──────────────────────────────────────────────────────────

@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    from pipeline.steps.step7_feedback_storage import store_feedback
    await run_in_threadpool(
        store_feedback,
        query=request.query,
        original_answer=request.original_answer,
        correction=request.correction,
    )
    return {"status": "saved"}


@app.post("/api/cache/save")
async def cache_save(request: CacheSaveRequest):
    from pipeline.utils.answer_cache import save_to_cache
    entry = await run_in_threadpool(
        save_to_cache,
        query=request.query,
        answer=request.answer,
        role=request.role,
        evidence=request.evidence,
        verification=request.verification,
        temporal_verification=request.temporal_verification,
    )
    return {"status": "saved", "id": entry["id"]}


@app.get("/api/cache")
async def cache_list():
    from pipeline.utils.answer_cache import list_cache
    return {"entries": list_cache()}


@app.delete("/api/cache/{entry_id}")
async def cache_delete(entry_id: str):
    from pipeline.utils.answer_cache import delete_from_cache
    deleted = await run_in_threadpool(delete_from_cache, entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    return {"status": "deleted"}


@app.get("/api/guidelines/{filename:path}")
async def get_guideline(filename: str):
    stem = filename.removesuffix(".pdf")
    matches = list(GUIDELINES_DIR.rglob(f"{stem}.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="Guideline not found")
    pdf_path = matches[0].resolve()
    if not str(pdf_path).startswith(str(GUIDELINES_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Cache-Control": "public, max-age=3600"},
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)
