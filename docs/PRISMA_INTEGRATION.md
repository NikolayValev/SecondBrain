# Prisma Integration for Second Brain

This document describes how to integrate a Next.js frontend with the Second Brain Python backend using Prisma ORM and PostgreSQL.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Second Brain System                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐     ┌─────────────────────────────────────────┐│
│  │   Python Backend    │     │           Next.js Frontend             ││
│  │   (FastAPI)         │     │                                         ││
│  │                     │     │  ┌───────────────┐  ┌────────────────┐ ││
│  │  ┌───────────────┐  │     │  │  React UI     │  │  API Routes    │ ││
│  │  │ File Watcher  │  │     │  │  Components   │  │  /api/*        │ ││
│  │  │ Indexer       │  │     │  └───────┬───────┘  └───────┬────────┘ ││
│  │  │ RAG Service   │  │     │          │                  │          ││
│  │  │ Inbox Proc.   │  │     │  ┌───────┴──────────────────┴────────┐ ││
│  │  └───────────────┘  │     │  │           Prisma Client           │ ││
│  │         │           │     │  └───────────────────────────────────┘ ││
│  │         ▼           │     │                    │                   ││
│  │  ┌───────────────┐  │     └────────────────────┼───────────────────┘│
│  │  │   SQLite DB   │  │                          │                    │
│  │  │   (Primary)   │  │                          │                    │
│  │  └───────┬───────┘  │                          │                    │
│  │          │          │                          │                    │
│  │    Sync Service     │                          │                    │
│  │          │          │                          │                    │
│  └──────────┼──────────┘                          │                    │
│             │                                     │                    │
│             ▼                                     ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       PostgreSQL Database                         │  │
│  │                        (Shared Data Store)                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup PostgreSQL Database

```bash
# Create database
createdb second_brain

# Or using Docker
docker run -d \
  --name second-brain-postgres \
  -e POSTGRES_DB=second_brain \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:16-alpine
```

### 2. Configure Python Backend

Add to your `.env` file:

```env
# PostgreSQL connection (matches Prisma convention)
DATABASE_URL=postgresql://postgres:password@localhost:5432/second_brain

# Enable sync features
POSTGRES_SYNC_ENABLED=true

# Optional: Sync on every file change (can be resource-intensive)
POSTGRES_SYNC_ON_CHANGE=false
```

### 3. Setup Next.js Frontend

```bash
# Create Next.js app
npx create-next-app@latest second-brain-ui --typescript --tailwind --app

# Install Prisma
cd second-brain-ui
npm install prisma @prisma/client
npx prisma init
```

### 4. Copy Prisma Schema

Copy `prisma/schema.prisma` from the Python project to your Next.js project's `prisma/` folder.

### 5. Generate Prisma Client and Apply Migrations

```bash
# Generate Prisma client
npx prisma generate

# Create and apply migration
npx prisma migrate dev --name init

# Verify tables created
npx prisma studio
```

### 6. Initial Sync

```bash
# Run from Python project
curl -X POST http://localhost:8000/sync -H "Content-Type: application/json" -d '{"mode": "full"}'
```

---

## Prisma Schema Reference

The schema is located at `prisma/schema.prisma`. Key models:

### Core Data Models

| Model | Description | Key Fields |
|-------|-------------|------------|
| `File` | Markdown files from vault | `path`, `title`, `content`, `mtime` |
| `Section` | Sections within files | `heading`, `level`, `content` |
| `Tag` | Unique tags | `name` |
| `Link` | Wikilinks between files | `from_file_id`, `to_path` |
| `Chunk` | Text chunks for embedding | `content`, `token_count` |
| `Embedding` | Vector embeddings | `embedding` (bytes), `dimensions` |

### Application Models

| Model | Description | Key Fields |
|-------|-------------|------------|
| `Conversation` | Chat sessions | `session_id`, `title` |
| `Message` | Chat messages | `role`, `content`, `sources` |
| `SearchHistory` | Search tracking | `query`, `results_count` |
| `SyncLog` | Sync history | `files_added`, `status` |
| `InboxProcessingLog` | Inbox processing history | `file_path`, `action` |

---

## API Endpoints Reference

### Sync Operations

#### Sync to PostgreSQL
```http
POST /sync
Content-Type: application/json

{
    "mode": "incremental"  // or "full"
}
```

**Response:**
```json
{
    "files_added": 10,
    "files_updated": 5,
    "files_deleted": 0,
    "sections": 150,
    "tags": 25,
    "links": 80,
    "chunks": 300,
    "embeddings": 300,
    "errors": [],
    "status": "completed"
}
```

#### Get PostgreSQL Stats
```http
GET /sync/stats
```

**Response:**
```json
{
    "file_count": 100,
    "section_count": 500,
    "tag_count": 50,
    "link_count": 200,
    "chunk_count": 1000,
    "embedding_count": 1000,
    "last_sync": "2024-01-15T10:30:00Z"
}
```

#### Sync Single File
```http
POST /sync/file?path=notes/my-note.md
```

### Conversation Operations

#### Create Conversation
```http
POST /conversations
Content-Type: application/json

{
    "session_id": "user-123",
    "title": "Research on AI"
}
```

#### Get Conversation
```http
GET /conversations/{id}
```

#### Add Message
```http
POST /conversations/{id}/messages
Content-Type: application/json

{
    "role": "user",
    "content": "What are the key concepts in my notes about machine learning?",
    "sources": null
}
```

#### List Conversations
```http
GET /conversations?session_id=user-123&limit=20
```

---

## Next.js Implementation Guide

### 1. Prisma Client Setup

Create `lib/prisma.ts`:

```typescript
import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: ['query'],
  })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma
```

### 2. API Routes

#### Search Files

Create `app/api/search/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const query = searchParams.get('q') || ''
  const limit = parseInt(searchParams.get('limit') || '20')

  if (!query) {
    return NextResponse.json({ results: [], count: 0 })
  }

  // Full-text search using PostgreSQL
  const results = await prisma.$queryRaw`
    SELECT 
      f.path,
      f.title,
      s.heading,
      LEFT(s.content, 200) as snippet
    FROM "Section" s
    JOIN "File" f ON f.id = s."fileId"
    WHERE to_tsvector('english', f.title || ' ' || s.heading || ' ' || s.content)
          @@ plainto_tsquery('english', ${query})
    ORDER BY ts_rank(
      to_tsvector('english', f.title || ' ' || s.heading || ' ' || s.content),
      plainto_tsquery('english', ${query})
    ) DESC
    LIMIT ${limit}
  `

  return NextResponse.json({
    results,
    count: (results as any[]).length,
    query
  })
}
```

#### List Files

Create `app/api/files/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const tag = searchParams.get('tag')
  const limit = parseInt(searchParams.get('limit') || '50')

  let files

  if (tag) {
    files = await prisma.file.findMany({
      where: {
        tags: {
          some: {
            tag: {
              name: tag
            }
          }
        }
      },
      include: {
        tags: {
          include: {
            tag: true
          }
        },
        _count: {
          select: {
            sections: true,
            outboundLinks: true
          }
        }
      },
      orderBy: { updatedAt: 'desc' },
      take: limit
    })
  } else {
    files = await prisma.file.findMany({
      include: {
        tags: {
          include: {
            tag: true
          }
        },
        _count: {
          select: {
            sections: true,
            outboundLinks: true
          }
        }
      },
      orderBy: { updatedAt: 'desc' },
      take: limit
    })
  }

  return NextResponse.json({
    files,
    count: files.length
  })
}
```

#### Get File Detail

Create `app/api/files/[id]/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const fileId = parseInt(params.id)

  const file = await prisma.file.findUnique({
    where: { id: fileId },
    include: {
      sections: true,
      tags: {
        include: {
          tag: true
        }
      },
      outboundLinks: true,
      chunks: {
        include: {
          embedding: true
        }
      }
    }
  })

  if (!file) {
    return NextResponse.json(
      { error: 'File not found' },
      { status: 404 }
    )
  }

  return NextResponse.json(file)
}
```

#### Tags API

Create `app/api/tags/route.ts`:

```typescript
import { NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET() {
  const tags = await prisma.tag.findMany({
    include: {
      _count: {
        select: {
          files: true
        }
      }
    },
    orderBy: {
      files: {
        _count: 'desc'
      }
    }
  })

  return NextResponse.json({
    tags: tags.map(t => ({
      id: t.id,
      name: t.name,
      count: t._count.files
    })),
    count: tags.length
  })
}
```

#### Conversations API

Create `app/api/conversations/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const sessionId = searchParams.get('sessionId')

  const conversations = await prisma.conversation.findMany({
    where: sessionId ? { sessionId } : undefined,
    include: {
      messages: {
        orderBy: { createdAt: 'asc' },
        take: 1  // Just get first message for preview
      },
      _count: {
        select: { messages: true }
      }
    },
    orderBy: { updatedAt: 'desc' },
    take: 20
  })

  return NextResponse.json({ conversations })
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  
  const conversation = await prisma.conversation.create({
    data: {
      sessionId: body.sessionId,
      title: body.title
    }
  })

  return NextResponse.json(conversation)
}
```

Create `app/api/conversations/[id]/messages/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const conversationId = parseInt(params.id)

  const messages = await prisma.message.findMany({
    where: { conversationId },
    orderBy: { createdAt: 'asc' }
  })

  return NextResponse.json({ messages })
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const conversationId = parseInt(params.id)
  const body = await request.json()

  const message = await prisma.message.create({
    data: {
      conversationId,
      role: body.role,
      content: body.content,
      sources: body.sources || null
    }
  })

  // Update conversation timestamp
  await prisma.conversation.update({
    where: { id: conversationId },
    data: { updatedAt: new Date() }
  })

  return NextResponse.json(message)
}
```

### 3. RAG Integration

To use the RAG feature, call the Python backend:

```typescript
// lib/secondBrain.ts

const SECOND_BRAIN_URL = process.env.SECOND_BRAIN_URL || 'http://localhost:8000'

export async function askQuestion(question: string): Promise<{
  answer: string
  sources: Array<{
    file_path: string
    file_title: string
    section: string | null
    similarity: number
  }>
}> {
  const response = await fetch(`${SECOND_BRAIN_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, include_sources: true })
  })

  if (!response.ok) {
    throw new Error(`RAG request failed: ${response.statusText}`)
  }

  return response.json()
}

export async function triggerSync(mode: 'full' | 'incremental' = 'incremental') {
  const response = await fetch(`${SECOND_BRAIN_URL}/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  })

  return response.json()
}
```

### 4. React Components

#### Search Component

```tsx
// components/Search.tsx
'use client'

import { useState } from 'react'

interface SearchResult {
  path: string
  title: string
  heading: string
  snippet: string
}

export function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`)
      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search your notes..."
          className="flex-1 px-4 py-2 border rounded-lg"
        />
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      <div className="space-y-2">
        {results.map((result, i) => (
          <div key={i} className="p-4 border rounded-lg">
            <h3 className="font-semibold">{result.title}</h3>
            <p className="text-sm text-gray-500">{result.heading}</p>
            <p className="text-sm mt-2">{result.snippet}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
```

#### Chat Interface

```tsx
// components/Chat.tsx
'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{
    file_path: string
    file_title: string
  }>
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      // Call Python backend for RAG
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage, include_sources: true })
      })

      const data = await response.json()
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }])
    } catch (error) {
      console.error('Chat error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.'
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[600px]">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-4 rounded-lg ${
              msg.role === 'user'
                ? 'bg-blue-100 ml-8'
                : 'bg-gray-100 mr-8'
            }`}
          >
            <p>{msg.content}</p>
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-2 text-sm text-gray-500">
                <p>Sources:</p>
                <ul className="list-disc list-inside">
                  {msg.sources.map((s, j) => (
                    <li key={j}>{s.file_title}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="bg-gray-100 p-4 rounded-lg mr-8 animate-pulse">
            Thinking...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your notes..."
          className="flex-1 px-4 py-2 border rounded-lg"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg"
        >
          Send
        </button>
      </form>
    </div>
  )
}
```

---

## Sync Strategies

### Strategy 1: Manual Sync (Recommended for Development)

Trigger sync manually or periodically:

```bash
# Full sync (clears and rebuilds)
curl -X POST http://localhost:8000/sync -d '{"mode": "full"}'

# Incremental sync (updates only changed files)
curl -X POST http://localhost:8000/sync -d '{"mode": "incremental"}'
```

### Strategy 2: Cron-based Sync

Add to `cron_jobs.py`:

```bash
# Run every 15 minutes
python cron_jobs.py sync --mode incremental
```

### Strategy 3: Real-time Sync

Set in `.env`:

```env
POSTGRES_SYNC_ON_CHANGE=true
```

This syncs each file to PostgreSQL when it's indexed. More resource-intensive but keeps data fresh.

---

## Environment Variables

### Python Backend (.env)

```env
# Required
VAULT_PATH=/path/to/obsidian/vault
DATABASE_URL=postgresql://user:pass@localhost:5432/second_brain

# Optional
POSTGRES_SYNC_ENABLED=true
POSTGRES_SYNC_ON_CHANGE=false

# LLM for RAG
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
```

### Next.js (.env.local)

```env
# Same DATABASE_URL for Prisma
DATABASE_URL=postgresql://user:pass@localhost:5432/second_brain

# Python backend URL for RAG
SECOND_BRAIN_URL=http://localhost:8000
```

---

## Troubleshooting

### Connection Issues

```bash
# Test PostgreSQL connection
psql $DATABASE_URL -c "SELECT 1"

# Test from Python
python -c "from app.db_postgres import get_postgres_db; import asyncio; asyncio.run(get_postgres_db().connect())"
```

### Sync Not Working

1. Verify `DATABASE_URL` is set
2. Check PostgreSQL is running
3. Ensure tables exist (run Prisma migration)
4. Check sync logs: `GET /sync/stats`

### Prisma Schema Changes

After modifying `schema.prisma`:

```bash
npx prisma migrate dev --name your_change_name
npx prisma generate
```

---

## Security Considerations

1. **Database Credentials**: Never commit `.env` files. Use secrets management in production.

2. **API Authentication**: Add authentication to both backends for production use.

3. **Network Isolation**: Keep PostgreSQL on internal network, not exposed to internet.

4. **CORS**: Configure FastAPI CORS for production:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://your-nextjs-app.com"],
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

---

## File Structure Summary

### Python Backend (new files)

```
SecondBrain/
├── app/
│   ├── db_postgres.py     # PostgreSQL database adapter
│   ├── sync_service.py    # SQLite to PostgreSQL sync
│   └── config.py          # Updated with POSTGRES_URL
├── prisma/
│   └── schema.prisma      # Prisma schema definition
└── requirements.txt       # Added asyncpg
```

### Next.js Frontend (create new)

```
second-brain-ui/
├── app/
│   ├── api/
│   │   ├── search/route.ts
│   │   ├── files/route.ts
│   │   ├── files/[id]/route.ts
│   │   ├── tags/route.ts
│   │   └── conversations/
│   │       ├── route.ts
│   │       └── [id]/messages/route.ts
│   ├── page.tsx
│   └── layout.tsx
├── components/
│   ├── Search.tsx
│   └── Chat.tsx
├── lib/
│   ├── prisma.ts
│   └── secondBrain.ts
├── prisma/
│   └── schema.prisma      # Copy from Python project
├── .env.local
└── package.json
```

---

## Summary

This integration allows:

1. **Python backend** continues to manage:
   - File watching and indexing
   - RAG and LLM operations
   - Inbox processing
   - SQLite as primary data store

2. **PostgreSQL** serves as:
   - Shared data layer between Python and Next.js
   - Prisma-compatible database for Next.js

3. **Next.js frontend** provides:
   - Modern React UI
   - Direct database access via Prisma
   - RAG queries via Python backend API
   - Conversation history management
