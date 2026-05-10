"""
memory.py — Pydantic schemas for Allora's three long-term memory types.

These schemas are used both by Trustcall (for structured extraction / patching)
and by the FastAPI layer (for serialization / validation).
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Profile Memory — who the user IS
# ---------------------------------------------------------------------------

class ProfileMemory(BaseModel):
    """
    Persistent social identity of the user.
    Populated progressively across many sessions.
    """

    interests: List[str] = Field(
        default_factory=list,
        description=(
            "Topics, activities or passions the user genuinely cares about. "
            "E.g. ['rock climbing', 'lo-fi music', 'cooking Thai food']."
        ),
    )
    personality_traits: List[str] = Field(
        default_factory=list,
        description=(
            "Observable personality characteristics. "
            "E.g. ['adventurous', 'empathetic', 'introvert-leaning', 'creative']."
        ),
    )
    social_style: Optional[str] = Field(
        default=None,
        description=(
            "How the user typically engages socially. "
            "E.g. 'prefers small gatherings over big parties', 'initiates plans often'."
        ),
    )
    vibe_summary: Optional[str] = Field(
        default=None,
        description=(
            "A one-or-two sentence human-readable 'vibe' that captures who this person is. "
            "E.g. 'Chill creative who loves deep conversations and spontaneous adventures'."
        ),
    )
    favorite_environments: List[str] = Field(
        default_factory=list,
        description=(
            "Places or settings where the user feels most alive or comfortable. "
            "E.g. ['coffee shops', 'hiking trails', 'rooftop bars']."
        ),
    )
    hobbies: List[str] = Field(
        default_factory=list,
        description=(
            "Specific hobbies or recurring activities. "
            "E.g. ['playing guitar', 'urban sketching', 'board games']."
        ),
    )
    dislikes: List[str] = Field(
        default_factory=list,
        description=(
            "Things the user dislikes, avoids, or considers turn-offs in social/dating context. "
            "E.g. ['smoking', 'loud clubs', 'chronic lateness']."
        ),
    )
    emotional_style: Optional[str] = Field(
        default=None,
        description=(
            "How the user tends to process or express emotions. "
            "E.g. 'openly expressive', 'processes internally first', 'humor as coping'."
        ),
    )


# ---------------------------------------------------------------------------
# 2. Context Memory — what's happening in the user's life RIGHT NOW
# ---------------------------------------------------------------------------

class ContextMemory(BaseModel):
    """
    Rolling window of recent life context.
    Updated per session; older entries may be pruned.
    """

    recent_topics: List[str] = Field(
        default_factory=list,
        description=(
            "Topics discussed in the most recent session(s). "
            "E.g. ['job change', 'moving to a new city', 'starting therapy']."
        ),
    )
    evolving_interests: List[str] = Field(
        default_factory=list,
        description=(
            "Interests the user has mentioned exploring recently but aren't yet settled. "
            "E.g. ['thinking about learning surfing', 'curious about meditation']."
        ),
    )
    recent_life_changes: List[str] = Field(
        default_factory=list,
        description=(
            "Significant recent transitions or events. "
            "E.g. ['just graduated', 'ended a long relationship', 'started remote work']."
        ),
    )
    recent_social_behavior: Optional[str] = Field(
        default=None,
        description=(
            "How the user has been socializing lately. "
            "E.g. 'going out less lately', 'spending more time with new friends from gym'."
        ),
    )
    current_mood_theme: Optional[str] = Field(
        default=None,
        description=(
            "The general emotional undertone across recent messages. "
            "E.g. 'energized and optimistic', 'going through a quieter phase'."
        ),
    )


# ---------------------------------------------------------------------------
# 3. Preference Memory — HOW the user likes to interact
# ---------------------------------------------------------------------------

class PreferenceMemory(BaseModel):
    """
    Meta-preferences about the conversation experience itself.
    Helps the agent adapt its style without being told explicitly.
    """

    conversation_style: Optional[str] = Field(
        default=None,
        description=(
            "The tone and rhythm the user responds best to. "
            "E.g. 'casual and witty', 'warm and reflective', 'direct and efficient'."
        ),
    )
    prefers_short_questions: bool = Field(
        default=False,
        description="True if the user has shown they prefer brief, one-topic questions.",
    )
    depth_preference: Optional[str] = Field(
        default=None,
        description=(
            "How deep the user wants questions to go. "
            "E.g. 'surface-level is fine', 'loves going deep on topics'."
        ),
    )
    sensitive_topics: List[str] = Field(
        default_factory=list,
        description=(
            "Topics the user has signaled they'd rather not discuss. "
            "E.g. ['family relationships', 'finances']."
        ),
    )
    response_length_preference: Optional[str] = Field(
        default=None,
        description=(
            "Whether the user prefers shorter or longer AI replies. "
            "E.g. 'short and punchy', 'likes detailed responses'."
        ),
    )
