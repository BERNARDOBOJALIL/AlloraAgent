"""
example_conversations.py
─────────────────────────
Runnable demo script that simulates two full sessions for one user.

Session 1: First encounter — interests, hobbies, vibe emerge naturally.
Session 2: New thread, same user — agent picks up where it left off.

Run with:
    cd allora_agent
    python docs/example_conversations.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import httpx

BASE_URL = "http://localhost:3000"
USER_ID = "alex_2025"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def chat(
    client: httpx.AsyncClient,
    thread_id: str,
    message: str,
    label: str = "",
) -> Dict[str, Any]:
    payload = {"user_id": USER_ID, "thread_id": thread_id, "message": message}
    resp = await client.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'─'*60}")
    if label:
        print(f"  [{label}]")
    print(f"  USER  : {message}")
    print(f"  ALLORA: {data['assistant_message']}")
    print(f"  Completion: {data['conversation_state']['profile_completion']*100:.0f}%")

    updates = data["memory_updates"]
    changed = []
    pm = updates["profile_memory"]
    if pm["interests"]:
        changed.append(f"interests+{pm['interests']}")
    if pm["traits"]:
        changed.append(f"traits+{pm['traits']}")
    if pm["social_style"]:
        changed.append(f"social_style='{pm['social_style']}'")
    if pm["vibe_summary"]:
        changed.append(f"vibe_summary='{pm['vibe_summary']}'")
    if pm["hobbies"]:
        changed.append(f"hobbies+{pm['hobbies']}")
    cm = updates["context_memory"]
    if cm["recent_topics"]:
        changed.append(f"topics+{cm['recent_topics']}")
    if cm["life_updates"]:
        changed.append(f"life+{cm['life_updates']}")
    pfm = updates["preference_memory"]
    if pfm["conversation_style"]:
        changed.append(f"conv_style='{pfm['conversation_style']}'")

    if changed:
        print(f"  Memory Δ : {' | '.join(changed)}")

    return data


async def print_profile(client: httpx.AsyncClient) -> None:
    resp = await client.get(f"{BASE_URL}/profile/{USER_ID}", timeout=10)
    resp.raise_for_status()
    profile = resp.json()
    print("\n" + "═"*60)
    print("  FULL PROFILE SNAPSHOT")
    print("═"*60)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print("═"*60)


# ---------------------------------------------------------------------------
# Session 1: Discovery
# ---------------------------------------------------------------------------

SESSION_1_TURNS = [
    # turn 1 — casual opener, agent asks first question
    ("Hey!", "opener"),
    # turn 2 — music / taste
    (
        "I've been really into making music lately, mostly lo-fi beats on my laptop. "
        "It started as a hobby but now I spend like 3 hours a day on it lol.",
        "music hobby",
    ),
    # turn 3 — social context
    (
        "I usually make music late at night, alone. I'm more of an introvert honestly, "
        "I recharge by having time for myself.",
        "introvert / nighttime",
    ),
    # turn 4 — interests branching
    (
        "Besides music I love going on long walks in the city at night, "
        "finding new coffee shops, and reading sci-fi.",
        "walks + reading",
    ),
    # turn 5 — life update
    (
        "I just moved to a new city like two months ago so I'm still finding my spots, "
        "but it's kind of exciting.",
        "life change",
    ),
    # turn 6 — emotional style
    (
        "I tend to overthink things a lot but once I'm comfortable with someone I open up super fast.",
        "emotional style",
    ),
]

# ---------------------------------------------------------------------------
# Session 2: Continuation (new thread, same user)
# ---------------------------------------------------------------------------

SESSION_2_TURNS = [
    # agent should acknowledge what it knows without being weird about it
    (
        "Hey, I'm back. Had kind of a rough week but I'm feeling better.",
        "returning user — mood check",
    ),
    # new interest surfaces
    (
        "I started going to a salsa class on Thursdays, my friend dragged me but it's actually fun.",
        "new hobby — salsa",
    ),
    # social style update
    (
        "It's funny, I used to avoid group stuff but lately I've been saying yes more.",
        "social style shift",
    ),
    # stop signal
    (
        "Alright that's probably enough about me for now haha, thanks!",
        "stop signal",
    ),
]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def main() -> None:
    async with httpx.AsyncClient() as client:

        # --- Verify server is up ---
        server_ready = False
        for attempt in range(1, 6):
            try:
                response = await client.get(f"{BASE_URL}/health", timeout=5)
                response.raise_for_status()
                server_ready = True
                break
            except (httpx.ConnectError, httpx.ReadTimeout):
                if attempt == 5:
                    print(
                        "❌  Server not ready or not responding. Start it with:\n"
                        "    uvicorn app.main:app --reload"
                    )
                    return
                await asyncio.sleep(1)

        if not server_ready:
            print("❌  Server not ready. Start it with:\n    uvicorn app.main:app --reload")
            return

        print("\n" + "█"*60)
        print("  SESSION 1 — First encounter")
        print("█"*60)

        for message, label in SESSION_1_TURNS:
            await chat(client, thread_id="session-1", message=message, label=label)
            await asyncio.sleep(0.5)

        await print_profile(client)

        print("\n\n" + "█"*60)
        print("  SESSION 2 — Returning user (new thread_id, same user_id)")
        print("█"*60)

        for message, label in SESSION_2_TURNS:
            result = await chat(client, thread_id="session-2", message=message, label=label)
            if not result["conversation_state"]["should_continue"]:
                print("\n  [Agent: profile conversation ended gracefully]")
                break
            await asyncio.sleep(0.5)

        await print_profile(client)


if __name__ == "__main__":
    asyncio.run(main())
