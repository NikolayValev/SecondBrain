"""Backfill the PostgreSQL 'embeddings' (bytes) table from SQLite."""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    from app.db import db
    from app.db_postgres import get_postgres_db

    db.initialize()
    pg = get_postgres_db()

    all_embs = db.get_all_embeddings()
    logger.info("SQLite embeddings to push: %d", len(all_embs))

    # Build chunk map: (file_path, chunk_index) -> pg chunk_id
    async with pg.acquire() as conn:
        rows = await conn.fetch(
            "SELECT c.id AS chunk_id, c.chunk_index, f.path AS file_path "
            "FROM chunks c JOIN files f ON f.id = c.file_id"
        )
    pg_lookup = {(r["file_path"], r["chunk_index"]): r["chunk_id"] for r in rows}

    inserted = 0
    skipped = 0
    for emb in all_embs:
        key = (emb["file_path"], emb["chunk_index"])
        pg_chunk_id = pg_lookup.get(key)
        if pg_chunk_id is None:
            skipped += 1
            continue

        async with pg.acquire() as conn:
            await conn.execute(
                "INSERT INTO embeddings (chunk_id, embedding, model, dimensions, created_at) "
                "VALUES ($1, $2, $3, $4, NOW()) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, "
                "model = EXCLUDED.model, "
                "dimensions = EXCLUDED.dimensions, "
                "created_at = NOW()",
                pg_chunk_id,
                emb["embedding"],
                "nomic-embed-text",
                emb["dimensions"],
            )
        inserted += 1

    logger.info("Inserted: %d, Skipped: %d", inserted, skipped)

    async with pg.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
        logger.info("PG embeddings table now has %d rows", count)

    await pg.close()


if __name__ == "__main__":
    asyncio.run(main())
