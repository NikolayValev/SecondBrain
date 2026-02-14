"""
Migration script: set up pgvector and push existing SQLite embeddings to PostgreSQL.

Usage:
    python run_pgvector_migration.py
"""

import asyncio
import logging
import struct
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    from app.config import Config

    if not Config.POSTGRES_URL:
        logger.error("No POSTGRES_URL / DATABASE_URL configured — aborting.")
        sys.exit(1)

    from app.db import db
    from app.db_postgres import get_postgres_db

    # ---- 1. Initialise local DB (no-op if already exists) ----
    db.initialize()

    # ---- 2. Read embeddings from SQLite ----
    all_embs = db.get_all_embeddings()
    if not all_embs:
        logger.warning("No embeddings in SQLite — nothing to migrate.")
        return

    dimensions = all_embs[0]["dimensions"]
    logger.info("Found %d embeddings in SQLite (dim=%d)", len(all_embs), dimensions)

    # ---- 3. Create pgvector extension + table ----
    pg = get_postgres_db()
    ok = await pg.ensure_pgvector(dimensions=dimensions)
    if not ok:
        logger.error(
            "pgvector setup failed. Make sure the 'vector' extension is "
            "available on your PostgreSQL server (e.g. Supabase, Neon, or "
            "'apt install postgresql-16-pgvector')."
        )
        sys.exit(1)

    # ---- 4. Ensure chunks exist in PG (they may already from sync) ----
    # We need the chunk rows referenced by FK. Check if the chunks table
    # has rows; if not, sync them first.
    async with pg.acquire() as conn:
        pg_chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    logger.info("PostgreSQL chunks table has %d rows", pg_chunk_count)

    if pg_chunk_count == 0:
        logger.info("Syncing files → sections → chunks to PostgreSQL first...")
        await _sync_core_data(db, pg)
        async with pg.acquire() as conn:
            pg_chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        logger.info("After sync: %d chunks in PostgreSQL", pg_chunk_count)

    # Build a mapping: sqlite chunk_id → pg chunk_id (by file_path + chunk_index)
    chunk_map = await _build_chunk_map(db, pg)
    logger.info("Mapped %d / %d chunk IDs between SQLite ↔ PG", len(chunk_map), len(all_embs))

    # ---- 5. Insert embeddings into embedding_vectors ----
    inserted = 0
    skipped = 0
    for emb in all_embs:
        sqlite_chunk_id = emb["chunk_id"]
        pg_chunk_id = chunk_map.get(sqlite_chunk_id)
        if pg_chunk_id is None:
            skipped += 1
            continue

        raw = emb["embedding"]  # bytes
        dims = emb["dimensions"]
        floats = list(struct.unpack(f"{dims}f", raw))
        vec_literal = "[" + ",".join(str(v) for v in floats) + "]"

        async with pg.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO embedding_vectors (chunk_id, embedding, model, dimensions, created_at)
                VALUES ($1, $2::vector, $3, $4, NOW())
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model     = EXCLUDED.model,
                    dimensions= EXCLUDED.dimensions,
                    created_at= NOW()
                """,
                pg_chunk_id,
                vec_literal,
                "nomic-embed-text",
                dims,
            )
        inserted += 1

    logger.info(
        "Migration complete: %d embeddings inserted, %d skipped (unmapped chunks)",
        inserted,
        skipped,
    )

    await pg.close()


async def _sync_core_data(db, pg) -> None:
    """Push files, sections, and chunks from SQLite → PostgreSQL."""
    all_files = db.get_all_files()
    for f in all_files:
        pg_file_id = await pg.upsert_file(
            path=f["path"], mtime=f["mtime"], title=f["title"], content=f["content"]
        )
        # Sections
        await pg.clear_file_sections(pg_file_id)
        sections = db.get_sections_by_file(f["id"])
        section_map: dict[int, int] = {}  # sqlite section id → pg section id
        for s in sections:
            pg_sec_id = await pg.add_section(
                file_id=pg_file_id,
                heading=s["heading"],
                level=s["level"],
                content=s["content"],
            )
            section_map[s["id"]] = pg_sec_id

        # Chunks
        await pg.clear_file_chunks(pg_file_id)
        chunks = db.get_chunks_by_file(f["id"])
        for c in chunks:
            pg_section_id = section_map.get(c["section_id"]) if c["section_id"] else None
            await pg.add_chunk(
                file_id=pg_file_id,
                section_id=pg_section_id,
                chunk_index=c["chunk_index"],
                content=c["content"],
                token_count=c["token_count"],
            )


async def _build_chunk_map(db, pg) -> dict[int, int]:
    """
    Build sqlite_chunk_id → pg_chunk_id mapping.
    
    Strategy: match on (file path, chunk_index).
    """
    # SQLite side: chunk_id → (file_path, chunk_index)
    all_embs_full = db.get_all_embeddings()  # has file_path, chunk_index, chunk_id
    sqlite_lookup: dict[tuple[str, int], int] = {}
    for e in all_embs_full:
        sqlite_lookup[(e["file_path"], e["chunk_index"])] = e["chunk_id"]

    # PG side: (file_path, chunk_index) → pg chunk id
    async with pg.acquire() as conn:
        rows = await conn.fetch("""
            SELECT c.id AS chunk_id, c.chunk_index, f.path AS file_path
            FROM chunks c
            JOIN files f ON f.id = c.file_id
        """)

    pg_lookup: dict[tuple[str, int], int] = {}
    for r in rows:
        pg_lookup[(r["file_path"], r["chunk_index"])] = r["chunk_id"]

    # Map
    result: dict[int, int] = {}
    for key, sqlite_cid in sqlite_lookup.items():
        pg_cid = pg_lookup.get(key)
        if pg_cid is not None:
            result[sqlite_cid] = pg_cid

    return result


if __name__ == "__main__":
    asyncio.run(main())
