-- Second Brain PostgreSQL Schema
-- Run this against your PostgreSQL database to create all tables

-- Files table (markdown files from vault)
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    mtime DOUBLE PRECISION NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);

-- Sections table (headings within files)
CREATE TABLE IF NOT EXISTS sections (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    heading TEXT NOT NULL,
    level INTEGER NOT NULL,
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sections_file_id ON sections(file_id);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

-- File-Tag junction table
CREATE TABLE IF NOT EXISTS file_tags (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    UNIQUE(file_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_file_tags_file_id ON file_tags(file_id);
CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id ON file_tags(tag_id);

-- Links table (wikilinks between files)
CREATE TABLE IF NOT EXISTS links (
    id SERIAL PRIMARY KEY,
    from_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    to_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_links_from_file_id ON links(from_file_id);
CREATE INDEX IF NOT EXISTS idx_links_to_path ON links(to_path);

-- Chunks table (text chunks for embeddings)
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);

-- Embeddings table (vector embeddings)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
    embedding BYTEA NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Metadata table (key-value store)
CREATE TABLE IF NOT EXISTS metadata (
    id SERIAL PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL
);

-- Sync logs table
CREATE TABLE IF NOT EXISTS sync_logs (
    id SERIAL PRIMARY KEY,
    synced_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    files_added INTEGER NOT NULL DEFAULT 0,
    files_updated INTEGER NOT NULL DEFAULT 0,
    files_deleted INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed',
    error TEXT,
    duration_ms INTEGER
);

-- Inbox processing logs
CREATE TABLE IF NOT EXISTS inbox_processing_logs (
    id SERIAL PRIMARY KEY,
    processed_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_path TEXT NOT NULL,
    source_path TEXT NOT NULL,
    destination_path TEXT,
    action TEXT NOT NULL,
    classification TEXT,
    tags_added TEXT,
    error TEXT
);

-- Search history (for analytics)
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results_count INTEGER NOT NULL,
    searched_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);

-- Conversations (for Next.js chat UI)
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    title TEXT,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);

-- Messages (chat messages)
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

-- Full-text search index on files and sections
CREATE INDEX IF NOT EXISTS idx_files_fts ON files USING GIN (to_tsvector('english', title || ' ' || content));
CREATE INDEX IF NOT EXISTS idx_sections_fts ON sections USING GIN (to_tsvector('english', heading || ' ' || content));

COMMENT ON TABLE files IS 'Markdown files from Obsidian vault';
COMMENT ON TABLE sections IS 'Sections/headings within files';
COMMENT ON TABLE chunks IS 'Text chunks for RAG embeddings';
COMMENT ON TABLE conversations IS 'Chat sessions for Next.js frontend';
