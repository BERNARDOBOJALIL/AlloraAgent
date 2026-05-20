# schemas/__init__.py
from .memory import ProfileMemory, ContextMemory, PreferenceMemory, MatchPreferenceMemory, Location
from .api import (
    ChatRequest,
    ChatResponse,
    MemoryUpdates,
    ProfileMemoryUpdate,
    ContextMemoryUpdate,
    PreferenceMemoryUpdate,
    ConversationState,
    FullProfileResponse,
    ProfileMemoryUpdateRequest,
    ProfileMemoryUpdateResponse,
    MatchPreferenceMemoryUpdateRequest,
    MatchPreferenceMemoryUpdateResponse,
    MatchPayloadResponse,
)

__all__ = [
    "ProfileMemory",
    "ContextMemory",
    "PreferenceMemory",
    "MatchPreferenceMemory",
    "Location",
    "ChatRequest",
    "ChatResponse",
    "MemoryUpdates",
    "ProfileMemoryUpdate",
    "ContextMemoryUpdate",
    "PreferenceMemoryUpdate",
    "ConversationState",
    "FullProfileResponse",
    "ProfileMemoryUpdateRequest",
    "ProfileMemoryUpdateResponse",
    "MatchPreferenceMemoryUpdateRequest",
    "MatchPreferenceMemoryUpdateResponse",
    "MatchPayloadResponse",
]
