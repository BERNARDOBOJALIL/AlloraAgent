# Allora — AI Profile-Building Agent

> A LangGraph + FastAPI agent that builds a rich social profile through emotionally intelligent conversation.

---

## What It Does

Allora's agent chats with users to learn who they are — their vibe, interests, social style, and personality — and builds a persistent profile that grows over time and across sessions.

It is **not** a generic chatbot. Every turn has a purpose: extract signal, update memory, and ask one good follow-up question.

The frontend already handles age, gender, relationship goals and profile settings.  
The agent handles everything else.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         FastAPI Layer                            │
│                                                                  │
│   POST /chat ──► AlloraGraph ──► ChatResponse (JSON)            │
│   GET  /profile/{user_id}  ──►  FullProfileResponse             │
│   DELETE /profile/{user_id}  ──► clear memory                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      LangGraph Pipeline                          │
│                                                                  │
│   START                                                          │
│     │                                                            │
│     ▼                                                            │
│  ┌──────────────┐                                                │
│  │ load_memories│  ← reads all 3 stores, formats context block  │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │ profile_agent│  ← Groq generates reply + JSON memory delta   │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                            │
│  │ extract_and_save │  ← validates, dedupes, persists, builds   │
│  └──────┬───────────┘    ChatResponse                           │
│         │                                                        │
│        END                                                       │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Memory Layer                                 │
│                                                                  │
│   ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│   │ Profile Memory │  │ Context Memory │  │ Preference Memory│  │
│   │                │  │                │  │                  │  │
│   │ interests      │  │ recent_topics  │  │ conv_style       │  │
│   │ traits         │  │ life_changes   │  │ short_questions  │  │
│   │ social_style   │  │ social_behavior│  │ depth_preference │  │
│   │ vibe_summary   │  │ mood_theme     │  │ sensitive_topics │  │
│   │ hobbies        │  │ evolving_ints  │  │                  │  │
│   │ fav_envs       │  └────────────────┘  └──────────────────┘  │
│   │ emotional_style│                                             │
│   └────────────────┘                                             │
│                                                                  │
│   LangGraph InMemoryStore (swap → Postgres/Redis in production)  │
└──────────────────────────────────────────────────────────────────┘
```

### Short-term vs Long-term Memory

| Type | Mechanism | Scope | Resets on? |
|---|---|---|---|
| Short-term | `MemorySaver` (checkpointer) | Conversation history per `thread_id` | New `thread_id` |
| Long-term | `InMemoryStore` (3 namespaces) | Profile per `user_id` | Never (until DELETE) |

---

## Project Structure

```
allora_agent/
├── app/
│   ├── main.py                    # FastAPI app + endpoints
│   ├── agent/
│   │   └── agent.py               # LangGraph graph, nodes, system prompt
│   ├── memory/
│   │   └── memory_manager.py      # Read/write/dedup across 3 stores
│   └── schemas/
│       ├── memory.py              # ProfileMemory, ContextMemory, PreferenceMemory
│       └── api.py                 # ChatRequest, ChatResponse, FullProfileResponse
├── docs/
│   └── example_conversations.py   # Runnable demo (2 sessions, 10 turns)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd allora_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (to use Groq / Llama 3.1)
```

### 3. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run example conversations

```bash
# In a separate terminal (server must be running)
python docs/example_conversations.py
```

### 4b. Run interactive console chat

```bash
# In a separate terminal (server must be running)
python docs/console_chat.py
```

### 5. Explore the interactive docs

```
http://localhost:8000/docs
```

---

## API Reference

### `POST /chat`

Send a user message and receive a structured response.

**Request**
```json
{
  "user_id": "user_abc123",
  "thread_id": "session-2025-05-09",
  "message": "I've been really into making music lately, mostly lo-fi beats."
}
```

**Response**
```json
{
  "assistant_message": "Lo-fi beats at midnight, I can picture it. What got you into making music instead of just listening?",
  "memory_updates": {
    "profile_memory": {
      "interests": ["lo-fi music"],
      "traits": [],
      "social_style": null,
      "vibe_summary": null,
      "favorite_environments": [],
      "hobbies": ["making lo-fi beats"],
      "emotional_style": null
    },
    "context_memory": {
      "recent_topics": ["music production"],
      "evolving_interests": [],
      "life_updates": [],
      "recent_social_behavior": null,
      "current_mood_theme": null
    },
    "preference_memory": {
      "conversation_style": null,
      "prefers_short_questions": false,
      "depth_preference": null,
      "sensitive_topics": []
    }
  },
  "conversation_state": {
    "profile_completion": 0.15,
    "should_continue": true,
    "turn_count": 1
  }
}
```

### `GET /profile/{user_id}`

Retrieve the full accumulated profile for a user.

### `DELETE /profile/{user_id}`

Wipe all stored memory for a user.

---

## Design Decisions

### Why three separate memory stores?

Each store has a different update cadence:
- **Profile**: accumulates slowly, almost never shrinks
- **Context**: rolling window, reflects current life phase
- **Preference**: meta-layer, informs *how* to run the conversation

Keeping them separate makes it easy to prune context without touching the profile, and to expose only certain data to the frontend.

### Why additive / patch-based updates?

Following the Trustcall philosophy from the baseline notebook: the agent returns *deltas*, not full replacements. The `MemoryManager` merges them, deduplicates lists, and only overwrites scalars when explicitly non-null. This prevents accidental data loss and hallucinated overwrites.

### Why one question per turn?

Asking multiple questions simultaneously feels like a form, not a conversation. A single thoughtful question per turn keeps the exchange feeling human and gives the user room to go deep rather than just ticking boxes.

---

## Swapping to a Persistent Database (Production)

Replace `InMemoryStore` in `memory_manager.py`:

```python
# Postgres example
from langgraph.store.postgres import PostgresStore

_store = PostgresStore(
    connection_string="postgresql://user:password@host:5432/allora"
)
memory_manager = MemoryManager(store=_store)
```

No other code changes needed — the rest of the system is store-agnostic.
