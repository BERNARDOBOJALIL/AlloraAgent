"""
api.py — Pydantic schemas for FastAPI request/response contracts.

Every endpoint returns a fully-typed, validated JSON structure.
"""

from __future__ import annotations

from typing import Any, Literal, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Partial memory-update snapshots returned in every ChatResponse
# (these reflect only what changed in this turn, not the full store)
# ---------------------------------------------------------------------------

class ProfileMemoryUpdate(BaseModel):
    interests: List[str] = Field(default_factory=list)
    traits: List[str] = Field(default_factory=list)
    social_style: Optional[str] = None
    vibe_summary: Optional[str] = None
    favorite_environments: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    emotional_style: Optional[str] = None


class ContextMemoryUpdate(BaseModel):
    recent_topics: List[str] = Field(default_factory=list)
    evolving_interests: List[str] = Field(default_factory=list)
    life_updates: List[str] = Field(default_factory=list)
    recent_social_behavior: Optional[str] = None
    current_mood_theme: Optional[str] = None


class PreferenceMemoryUpdate(BaseModel):
    conversation_style: Optional[str] = None
    prefers_short_questions: bool = False
    depth_preference: Optional[str] = None
    sensitive_topics: List[str] = Field(default_factory=list)


class MemoryUpdates(BaseModel):
    """Snapshot of memory changes that occurred in this conversation turn."""
    profile_memory: ProfileMemoryUpdate = Field(default_factory=ProfileMemoryUpdate)
    context_memory: ContextMemoryUpdate = Field(default_factory=ContextMemoryUpdate)
    preference_memory: PreferenceMemoryUpdate = Field(default_factory=PreferenceMemoryUpdate)


class ConversationState(BaseModel):
    """Metadata about the overall profile-building progress."""
    profile_completion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction [0–1] of profile fields populated.",
    )
    should_continue: bool = Field(
        default=True,
        description="False when the user signals they want to stop or the profile is complete.",
    )
    turn_count: int = Field(default=0, description="Total turns in this session.")


# ---------------------------------------------------------------------------
# Main request / response shapes
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Sent by the frontend each time the user sends a message.
    """
    user_id: str = Field(
        ...,
        description="Unique, stable user identifier from the frontend auth system.",
    )
    thread_id: str = Field(
        ...,
        description=(
            "Session / conversation identifier. "
            "A new thread_id resets short-term memory while preserving long-term stores."
        ),
    )
    message: str = Field(..., description="The user's raw message text.")


class ChatResponse(BaseModel):
    """
    Returned after each conversation turn.
    Mirrors the structured JSON schema required by the specification.
    """
    assistant_message: str = Field(
        ...,
        description="The natural-language reply shown to the user.",
    )
    memory_updates: MemoryUpdates = Field(
        default_factory=MemoryUpdates,
        description="Delta of memory changes that happened in this turn.",
    )
    conversation_state: ConversationState = Field(
        default_factory=ConversationState,
        description="Progress and control signals for the frontend.",
    )


ProfileEditCategory = Literal[
    "interests",
    "personality_traits",
    "traits",
    "social_style",
    "vibe_summary",
    "favorite_environments",
    "hobbies",
    "dislikes",
    "emotional_style",
]


class ProfileCategoryUpdateRequest(BaseModel):
    """
    Direct edit for one profile category. The user's text is formatted only
    for the selected category and does not update any other memory field.
    """
    text: str = Field(..., min_length=1, description="Raw user text for this category.")


class ProfileCategoryUpdateResponse(BaseModel):
    user_id: str
    category: str
    formatted_value: Any
    profile_memory: dict = Field(default_factory=dict)
    profile_completion: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Full profile retrieval
# ---------------------------------------------------------------------------

class FullProfileResponse(BaseModel):
    """
    Returned by GET /profile/{user_id}.
    Contains the aggregated, deduplicated long-term memory.
    """
    user_id: str
    profile_memory: dict = Field(default_factory=dict)
    context_memory: dict = Field(default_factory=dict)
    preference_memory: dict = Field(default_factory=dict)
    profile_completion: float = Field(default=0.0, ge=0.0, le=1.0)
