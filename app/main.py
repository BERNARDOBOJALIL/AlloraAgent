"""
main.py — Allora Profile Agent API

Endpoints
─────────
POST /chat              → Send a message, get a structured response + memory updates
GET  /profile/{user_id} → Retrieve the full accumulated profile
DELETE /profile/{user_id} → Clear all memory for a user (useful for testing)
GET  /health            → Basic health check
"""

from __future__ import annotations

import re
import unicodedata

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent.agent import graph
from app.memory.memory_manager import memory_manager
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    FullProfileResponse,
    ProfileCategoryUpdateRequest,
    ProfileCategoryUpdateResponse,
    ProfileEditCategory,
)

app = FastAPI(
    title="Allora Profile Agent",
    description=(
        "AI-powered social profile builder. "
        "Learns who a user is through natural conversation and builds "
        "a rich, persistent profile across sessions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the profile agent",
    response_description="Agent reply + memory deltas + conversation state",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    The frontend calls this endpoint each time the user sends a message.

    - `user_id` identifies the user across sessions (long-term memory key).
    - `thread_id` identifies the current conversation (short-term memory key).
      Using a new `thread_id` resets the conversation context while preserving
      the user's accumulated profile.
    - `message` is the raw user text.
    """
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "memory_context": "",
        "raw_agent_output": "",
        "chat_response": None,
    }

    # Run the LangGraph pipeline
    result = await graph.ainvoke(initial_state, config=config)

    chat_response: ChatResponse | None = result.get("chat_response")

    if chat_response is None:
        raise HTTPException(
            status_code=500,
            detail="Agent failed to produce a response. Check server logs.",
        )

    return chat_response


def _clean_profile_edit_item(item: str) -> str:
    item = item.strip()
    item = re.sub(r"^[\W_]+|[\W_]+$", "", item)
    item = re.sub(
        r"^(me gusta(?:n)?|me encanta(?:n)?|prefiero|busco|quiero|valoro|soy|me considero|i like|i love|i prefer)\s+",
        "",
        item,
        flags=re.I,
    )
    item = re.sub(r"\s+", " ", item).strip()
    return item.lower()


def _split_profile_edit_items(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\b(?:y tambien|y también|tambien|también|ademas|además|also)\b", ",", text, flags=re.I)
    raw_items = re.split(r",|;|\n|/| & | y | e | and ", text, flags=re.I)
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _clean_profile_edit_item(raw_item)
        if item and item not in seen:
            items.append(item)
            seen.add(item)
    return items


def _format_profile_edit_sentence(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _canonical_profile_category(category: str) -> str:
    return "personality_traits" if category == "traits" else category


def _format_profile_category_value(category: str, text: str) -> str | list[str]:
    list_fields = {"interests", "personality_traits", "favorite_environments", "hobbies", "dislikes"}
    if category in list_fields:
        return _split_profile_edit_items(text)
    return _format_profile_edit_sentence(text)


@app.patch(
    "/profile/{user_id}/profile-memory/{category}",
    response_model=ProfileCategoryUpdateResponse,
    summary="Edit one profile-memory category directly",
)
async def update_profile_memory_category(
    user_id: str,
    category: ProfileEditCategory,
    request: ProfileCategoryUpdateRequest,
) -> ProfileCategoryUpdateResponse:
    """
    Replaces exactly one profile category using only the text provided by the user.
    This endpoint is intended for explicit profile-edit screens, not conversation.
    """
    field = _canonical_profile_category(category)
    formatted_value = _format_profile_category_value(field, request.text)

    if formatted_value == "" or formatted_value == []:
        raise HTTPException(status_code=400, detail="No usable profile content was provided.")

    try:
        updated_profile = memory_manager.replace_profile_field(user_id, field, formatted_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ProfileCategoryUpdateResponse(
        user_id=user_id,
        category=field,
        formatted_value=formatted_value,
        profile_memory=updated_profile.model_dump(),
        profile_completion=memory_manager.compute_completion(user_id),
    )


# ---------------------------------------------------------------------------
# GET /profile/{user_id}
# ---------------------------------------------------------------------------

@app.get(
    "/profile/{user_id}",
    response_model=FullProfileResponse,
    summary="Get a user's full accumulated profile",
)
async def get_profile(user_id: str) -> FullProfileResponse:
    """
    Returns all three memory stores merged into one structured document.
    Useful for displaying a profile summary in the app or debugging.
    """
    all_memory = memory_manager.get_all(user_id)
    completion = memory_manager.compute_completion(user_id)

    return FullProfileResponse(
        user_id=user_id,
        profile_completion=completion,
        **all_memory,
    )


# ---------------------------------------------------------------------------
# DELETE /profile/{user_id}
# ---------------------------------------------------------------------------

@app.delete(
    "/profile/{user_id}",
    summary="Clear all stored memory for a user",
    response_description="Confirmation message",
)
async def delete_profile(user_id: str) -> dict:
    """
    Wipes the long-term memory for the given user.
    Useful during testing or when a user requests data deletion.
    """
    store = memory_manager.store
    for namespace_prefix in ["profile", "context", "preference"]:
        ns = (namespace_prefix, user_id)
        items = store.search(ns)
        for item in items:
            store.delete(ns, item.key)

    return {"message": f"All memory for user '{user_id}' has been cleared."}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "allora-profile-agent"}
