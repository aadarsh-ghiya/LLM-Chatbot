import json
import os
import secrets
import shutil
import time
from hashlib import sha256
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
WEB_DATA_DIR = BASE_DIR / "web_data" / "sessions"
CACHE_DIR = BASE_DIR / "web_data" / "cache"
ALLOWED_EXTENSIONS = {".pdf"}

CHAIN_CACHE = {}
STATE_CACHE = {}
MESSAGE_CACHE = {}
ACTIVE_UPLOADS = set()


def get_process_pdfs():
    from extract_and_chunk import process_pdfs

    return process_pdfs


def get_chunk_helpers():
    from extract_and_chunk import clean_text, extract_text_from_pdf, list_pdfs, split_into_chunks

    return list_pdfs, extract_text_from_pdf, clean_text, split_into_chunks


def get_rag_helpers():
    from conversation_chain import (
        ask_question,
        build_conversation_chain,
        build_conversation_chain_from_vector_store,
        create_vector_store_from_chunks,
    )

    return (
        ask_question,
        build_conversation_chain,
        create_vector_store_from_chunks,
        build_conversation_chain_from_vector_store,
    )


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "lab9-web-ui")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @app.before_request
    def ensure_session_id():
        session.permanent = True
        get_session_id()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def get_state():
        session_id = get_session_id()
        state = normalize_state(session_id, load_state(session_id))
        return jsonify(
            {
                "state": state,
                "messages": load_messages(session_id),
            }
        )

    @app.post("/api/upload")
    def upload():
        session_id = get_session_id()
        files = [file for file in request.files.getlist("pdfs") if file and file.filename]
        if not files:
            return jsonify({"error": "Select one or more PDF files to analyze."}), 400

        paths = reset_session_workspace(session_id)
        saved_files = []
        rejected_files = []

        for file in files:
            filename = secure_filename(file.filename)
            if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                rejected_files.append(file.filename)
                continue

            destination = unique_path(paths["input"] / filename)
            file.save(destination)
            saved_files.append(destination.name)

        if not saved_files:
            return jsonify({"error": "Only PDF files are supported."}), 400

        bundle_key = build_bundle_key([paths["input"] / file_name for file_name in saved_files])
        save_messages(session_id, [])
        state = {
            "ready": False,
            "processing": True,
            "indexed_files": saved_files,
            "rejected_files": rejected_files,
            "pdf_count": 0,
            "chunk_count": 0,
            "status": f"Upload received. Preparing analysis for {len(saved_files)} PDF(s)...",
            "updated_at": timestamp_now(),
        }
        save_state(session_id, state)
        print(f"[upload] Session {session_id}: received {len(saved_files)} file(s)", flush=True)

        cached_state = try_restore_cached_index(session_id, bundle_key, rejected_files)
        if cached_state:
            print(f"[upload] Session {session_id}: cache hit for uploaded PDF set", flush=True)
            return jsonify({"state": cached_state, "messages": []}), 200

        ACTIVE_UPLOADS.add(session_id)
        try:
            analyze_uploaded_pdfs(session_id, bundle_key)
        finally:
            ACTIVE_UPLOADS.discard(session_id)

        final_state = normalize_state(session_id, load_state(session_id))
        status_code = 200 if final_state.get("ready") else 500
        payload = {"state": final_state, "messages": []}
        if status_code != 200:
            payload["error"] = final_state.get("status") or "Analysis failed."
        return jsonify(payload), status_code

    @app.post("/api/chat")
    def chat():
        session_id = get_session_id()
        payload = request.get_json(silent=True) or {}
        question = (payload.get("message") or "").strip()
        if not question:
            return jsonify({"error": "Enter a question before sending."}), 400

        state = normalize_state(session_id, load_state(session_id))
        if state.get("processing"):
            return jsonify({"error": "PDF analysis is still running. Please wait until it finishes."}), 400
        if not state.get("ready"):
            return jsonify({"error": "Upload and analyze PDFs before starting the chat."}), 400

        try:
            ask_question, _, _, _ = get_rag_helpers()
            chain = get_or_create_chain(session_id)
            answer = ask_question(chain, question)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        messages = load_messages(session_id)
        messages.extend(
            [
                {"role": "user", "text": question, "timestamp": timestamp_now()},
                {"role": "assistant", "text": answer, "timestamp": timestamp_now()},
            ]
        )
        save_messages(session_id, messages)
        return jsonify({"messages": messages, "answer": answer})

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "Uploaded files are too large. Try a smaller PDF or fewer files."}), 413

    @app.errorhandler(Exception)
    def handle_exception(error):
        if isinstance(error, HTTPException):
            if request.path.startswith("/api/"):
                return jsonify({"error": error.description}), error.code
            return error

        if request.path.startswith("/api/"):
            return jsonify({"error": str(error)}), 500
        return "Internal Server Error", 500

    return app


def get_session_id() -> str:
    if "chat_session_id" not in session:
        session["chat_session_id"] = secrets.token_hex(16)
    return session["chat_session_id"]


def session_paths(session_id: str) -> dict:
    root = WEB_DATA_DIR / session_id
    return {
        "root": root,
        "input": root / "input_pdf",
        "output": root / "output",
        "vector_store": root / "vector_store",
        "state_file": root / "state.json",
        "messages_file": root / "messages.json",
    }


def reset_session_workspace(session_id: str) -> dict:
    paths = session_paths(session_id)
    if paths["root"].exists():
        shutil.rmtree(paths["root"])

    paths["input"].mkdir(parents=True, exist_ok=True)
    paths["output"].mkdir(parents=True, exist_ok=True)
    paths["vector_store"].mkdir(parents=True, exist_ok=True)

    CHAIN_CACHE.pop(session_id, None)
    STATE_CACHE.pop(session_id, None)
    MESSAGE_CACHE.pop(session_id, None)
    return paths


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def analyze_uploaded_pdfs(session_id: str, bundle_key: str | None = None):
    paths = session_paths(session_id)

    try:
        update_state(session_id, status="Preparing analysis...", processing=True, ready=False)
        list_pdfs, extract_text_from_pdf, clean_text, split_into_chunks = get_chunk_helpers()

        update_state(
            session_id,
            status="Extracting text and creating chunks...",
            processing=True,
            ready=False,
        )
        pdf_paths = list_pdfs(str(paths["input"]))
        num_files = len(pdf_paths)
        total_chunks = 0
        chunks = []
        metadatas = []

        for pdf_path in pdf_paths:
            file_id = Path(pdf_path).stem
            print(f"[upload] Session {session_id}: extracting text from {Path(pdf_path).name}", flush=True)
            raw_text = extract_text_from_pdf(pdf_path)
            print(f"[upload] Session {session_id}: splitting chunks for {Path(pdf_path).name}", flush=True)
            cleaned_text = clean_text(raw_text)
            file_chunks = split_into_chunks(cleaned_text)
            print(
                f"[upload] Session {session_id}: created {len(file_chunks)} chunks from {Path(pdf_path).name}",
                flush=True,
            )

            total_chunks += len(file_chunks)
            for index, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                metadatas.append({"file_id": file_id, "chunk_index": index})

        update_state(
            session_id,
            pdf_count=num_files,
            chunk_count=total_chunks,
            status=f"Created {total_chunks} chunks. Building embeddings and vector store...",
            processing=True,
            ready=False,
        )

        print(f"[upload] Session {session_id}: loading AI dependencies", flush=True)
        (
            _,
            _build_conversation_chain,
            create_vector_store_from_chunks,
            build_conversation_chain_from_vector_store,
        ) = get_rag_helpers()
        print(f"[upload] Session {session_id}: creating embeddings and FAISS index", flush=True)
        vector_store = create_vector_store_from_chunks(
            chunks=chunks,
            metadatas=metadatas,
            out_folder=str(paths["vector_store"]),
        )
        chain = build_conversation_chain_from_vector_store(vector_store)

        CHAIN_CACHE[session_id] = chain
        update_state(
            session_id,
            ready=True,
            processing=False,
            status=f"Indexed {num_files} PDF(s) into {total_chunks} chunks.",
            updated_at=timestamp_now(),
        )
        print(
            f"[upload] Session {session_id}: indexed {num_files} PDF(s) into {total_chunks} chunks",
            flush=True,
        )
        if bundle_key:
            save_cached_index(
                bundle_key=bundle_key,
                vector_store_folder=paths["vector_store"],
                state={
                    "indexed_files": load_state(session_id).get("indexed_files", []),
                    "pdf_count": num_files,
                    "chunk_count": total_chunks,
                },
            )
    except Exception as exc:
        CHAIN_CACHE.pop(session_id, None)
        update_state(
            session_id,
            ready=False,
            processing=False,
            status=f"Analysis failed: {exc}",
            updated_at=timestamp_now(),
        )
        print(f"[upload] Session {session_id}: failed - {exc}", flush=True)


def get_or_create_chain(session_id: str):
    if session_id in CHAIN_CACHE:
        return CHAIN_CACHE[session_id]

    paths = session_paths(session_id)
    state = load_state(session_id)
    if not state.get("ready"):
        raise RuntimeError("No indexed PDFs are available yet.")

    _, build_conversation_chain, _, _ = get_rag_helpers()
    chain = build_conversation_chain(
        db_path=str(paths["output"] / "chunks.db"),
        vector_store_folder=str(paths["vector_store"]),
        rebuild_vector_store=False,
    )

    for message in load_messages(session_id):
        if message["role"] == "user":
            chain.memory.chat_memory.add_user_message(message["text"])
        elif message["role"] == "assistant":
            chain.memory.chat_memory.add_ai_message(message["text"])

    CHAIN_CACHE[session_id] = chain
    return chain


def load_state(session_id: str) -> dict:
    if session_id in STATE_CACHE:
        return STATE_CACHE[session_id]

    paths = session_paths(session_id)
    default_state = {
        "ready": False,
        "processing": False,
        "indexed_files": [],
        "rejected_files": [],
        "pdf_count": 0,
        "chunk_count": 0,
        "status": "Upload one or more PDFs to start building the knowledge base.",
        "updated_at": None,
    }
    state = read_json(paths["state_file"], default_state)
    STATE_CACHE[session_id] = state
    return state


def normalize_state(session_id: str, state: dict) -> dict:
    if state.get("processing") and session_id not in ACTIVE_UPLOADS:
        normalized = state.copy()
        normalized["processing"] = False
        normalized["ready"] = False
        normalized["status"] = "Previous analysis did not finish. Please upload the PDF again."
        normalized["updated_at"] = timestamp_now()
        save_state(session_id, normalized)
        return normalized

    return state


def cache_paths(bundle_key: str) -> dict:
    root = CACHE_DIR / bundle_key
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "vector_store": root / "vector_store",
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle_key(files: list[Path]) -> str:
    payload = [
        {
            "name": file.name,
            "size": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in sorted(files, key=lambda item: item.name)
    ]
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def try_restore_cached_index(session_id: str, bundle_key: str, rejected_files: list[str]) -> dict | None:
    paths = session_paths(session_id)
    cache = cache_paths(bundle_key)
    if not cache["manifest"].exists() or not cache["vector_store"].exists():
        return None

    manifest = read_json(cache["manifest"], None)
    if not manifest:
        return None

    if paths["vector_store"].exists():
        shutil.rmtree(paths["vector_store"])
    shutil.copytree(cache["vector_store"], paths["vector_store"])
    CHAIN_CACHE.pop(session_id, None)

    state = {
        "ready": True,
        "processing": False,
        "indexed_files": manifest.get("indexed_files", []),
        "rejected_files": rejected_files,
        "pdf_count": manifest.get("pdf_count", 0),
        "chunk_count": manifest.get("chunk_count", 0),
        "status": f"Loaded cached index for {manifest.get('pdf_count', 0)} PDF(s).",
        "updated_at": timestamp_now(),
    }
    save_state(session_id, state)
    return state


def save_cached_index(bundle_key: str, vector_store_folder: Path, state: dict):
    cache = cache_paths(bundle_key)
    if cache["root"].exists():
        return

    cache["root"].mkdir(parents=True, exist_ok=True)
    shutil.copytree(vector_store_folder, cache["vector_store"])
    write_json(cache["manifest"], state)


def save_state(session_id: str, state: dict):
    paths = session_paths(session_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_json(paths["state_file"], state)
    STATE_CACHE[session_id] = state


def update_state(session_id: str, **updates):
    state = load_state(session_id).copy()
    state.update(updates)
    save_state(session_id, state)


def load_messages(session_id: str) -> list:
    if session_id in MESSAGE_CACHE:
        return MESSAGE_CACHE[session_id]

    paths = session_paths(session_id)
    messages = read_json(paths["messages_file"], [])
    MESSAGE_CACHE[session_id] = messages
    return messages


def save_messages(session_id: str, messages: list):
    paths = session_paths(session_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_json(paths["messages_file"], messages)
    MESSAGE_CACHE[session_id] = messages


def read_json(path: Path, fallback):
    if not path.exists():
        return fallback

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def timestamp_now() -> str:
    return datetime.now().strftime("%b %d, %Y %I:%M %p")


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5050"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(f"[startup] Launching Flask server on http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=debug, threaded=False)
