import os
import argparse
import json
import sqlite3
import pdfplumber

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from langchain_text_splitters import CharacterTextSplitter

def list_pdfs(folder: str) -> List[str]:
    p = Path(folder)
    return sorted([str(x) for x in p.glob("**/*.pdf")])

def extract_text_from_pdf(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
    return "\n\n".join(texts)

def clean_text(raw: str) -> str:
    text = raw.replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None) -> List[str]:
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = CharacterTextSplitter(
        separator=separators[0],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = splitter.split_text(text)
    return chunks

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_text_file(text: str, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_chunks_json(chunks: List[str], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

def save_chunks_to_sqlite(db_path: str, file_id: str, chunks: List[str]):
    ensure_dir(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT,
            chunk_index INTEGER,
            text TEXT
        );
    """)

    cur.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    for i, chunk in enumerate(chunks):
        cur.execute("INSERT INTO chunks (file_id, chunk_index, text) VALUES (?, ?, ?)",
                    (file_id, i, chunk))
    conn.commit()
    conn.close()

def process_pdfs(
    input_folder: str,
    output_folder: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    persist_sqlite: bool = True
) -> Tuple[int, int]:
    pdf_paths = list_pdfs(input_folder)
    if not pdf_paths:
        print(f"No PDFs found in {input_folder}")
        return 0, 0

    texts_out_dir = os.path.join(output_folder, "texts")
    chunks_out_dir = os.path.join(output_folder, "chunks")
    db_path = os.path.join(output_folder, "chunks.db")

    total_chunks = 0
    ensure_dir(texts_out_dir)
    ensure_dir(chunks_out_dir)

    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        filename = Path(pdf_path).stem
        raw = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw)

        text_file_path = os.path.join(texts_out_dir, f"{filename}.txt")
        save_text_file(cleaned, text_file_path)

        chunks = split_into_chunks(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)

        chunks_json_path = os.path.join(chunks_out_dir, f"{filename}_chunks.json")
        save_chunks_json(chunks, chunks_json_path)

        if persist_sqlite:
            save_chunks_to_sqlite(db_path, file_id=filename, chunks=chunks)

    return len(pdf_paths), total_chunks

def parse_args():
    parser = argparse.ArgumentParser(description="Extract and chunk PDFs for embeddings.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing PDFs")
    parser.add_argument("--output_folder", type=str, default="./output", help="Where to save texts & chunks")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size in characters (default 500)")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap in characters")
    parser.add_argument("--no_sqlite", action="store_true", help="Disable saving chunks to sqlite")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    num_files, total_chunks = process_pdfs(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_sqlite=not args.no_sqlite
    )
    print(f"Processed {num_files} PDF(s). Total chunks created: {total_chunks}")
    print(f"Outputs written to: {os.path.abspath(args.output_folder)}")