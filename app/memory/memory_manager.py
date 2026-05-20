"""
memory_manager.py — Persistent memory layer for Allora's profile agent.

Wraps LangGraph's InMemoryStore (swap for Redis/Postgres in production)
and exposes clean read/write helpers for each of the three memory namespaces.
Handles deduplication so the agent never writes duplicate interests or traits.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

from langgraph.store.memory import InMemoryStore

from app.memory.mongo_store import MongoMemoryStore
from app.schemas.memory import ProfileMemory, ContextMemory, PreferenceMemory, MatchPreferenceMemory


# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------
NS_PROFILE = "profile"
NS_CONTEXT = "context"
NS_PREFERENCE = "preference"
NS_MATCH_PREFERENCE = "match_preference"

# Single-document key per user in each namespace
DOC_KEY = "data"


class MemoryStore(Protocol):
    def search(self, namespace: Tuple[str, str]) -> List[Any]:
        ...

    def put(self, namespace: Tuple[str, str], key: str, value: Dict[str, Any]) -> None:
        ...

    def delete(self, namespace: Tuple[str, str], key: str) -> None:
        ...


class MemoryManager:
    """
    Centralised access to Allora's three long-term memory stores.

    Set MONGODB_URI (or MONGO_URI) to persist memory in MongoDB Atlas.
    Without that env var, local development uses LangGraph's InMemoryStore.
    """

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self.store: MemoryStore = store or InMemoryStore()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _namespace(self, kind: str, user_id: str) -> Tuple[str, str]:
        return (kind, user_id)

    def _read_raw(self, kind: str, user_id: str) -> Optional[Dict[str, Any]]:
        ns = self._namespace(kind, user_id)
        results = self.store.search(ns)
        if not results:
            return None
        return results[0].value  # single-document pattern

    def _write_raw(self, kind: str, user_id: str, data: Dict[str, Any]) -> None:
        ns = self._namespace(kind, user_id)
        # Always upsert the same key so we maintain a single document per user
        self.store.put(ns, DOC_KEY, data)

    @staticmethod
    def _dedupe_list(existing: List[str], incoming: List[str]) -> List[str]:
        """Merge two lists, case-insensitively deduplicating."""
        seen = {item.lower() for item in existing}
        merged = list(existing)
        for item in incoming:
            if item.lower() not in seen:
                merged.append(item)
                seen.add(item.lower())
        return merged

    # ------------------------------------------------------------------
    # Profile Memory
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> ProfileMemory:
        raw = self._read_raw(NS_PROFILE, user_id)
        if raw is None:
            return ProfileMemory()
        return ProfileMemory(**raw)

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> ProfileMemory:
        """
        Merge `updates` into the existing profile.
        List fields are deduped; scalar fields are overwritten only if non-None.
        """
        current = self.get_profile(user_id)
        merged = current.model_dump()

        list_fields = {"interests", "personality_traits", "favorite_environments", "hobbies", "dislikes"}
        scalar_fields = {"edad", "genero", "bio", "social_style", "vibe_summary", "emotional_style", "location"}

        for field, value in updates.items():
            if field not in merged:
                continue
            if field in list_fields and value:
                merged[field] = self._dedupe_list(merged[field], value)
            elif field in scalar_fields and value is not None:
                merged[field] = value

        result = ProfileMemory(**merged)
        self._write_raw(NS_PROFILE, user_id, result.model_dump())
        return result

    def replace_profile_field(self, user_id: str, field: str, value: Any) -> ProfileMemory:
        """Replace exactly one profile field with a pre-formatted value."""
        current = self.get_profile(user_id)
        merged = current.model_dump()
        if field not in merged:
            raise ValueError(f"Unknown profile field: {field}")

        list_fields = {"interests", "personality_traits", "favorite_environments", "hobbies", "dislikes"}
        scalar_fields = {"edad", "genero", "bio", "social_style", "vibe_summary", "emotional_style", "location"}

        if field in list_fields:
            merged[field] = self._dedupe_list([], value or [])
        elif field in scalar_fields:
            merged[field] = value or None

        result = ProfileMemory(**merged)
        self._write_raw(NS_PROFILE, user_id, result.model_dump())
        return result

    # ------------------------------------------------------------------
    # Context Memory
    # ------------------------------------------------------------------

    def get_context(self, user_id: str) -> ContextMemory:
        raw = self._read_raw(NS_CONTEXT, user_id)
        if raw is None:
            return ContextMemory()
        return ContextMemory(**raw)

    def update_context(self, user_id: str, updates: Dict[str, Any]) -> ContextMemory:
        current = self.get_context(user_id)
        merged = current.model_dump()

        list_fields = {"recent_topics", "evolving_interests", "recent_life_changes"}
        scalar_fields = {"recent_social_behavior", "current_mood_theme"}

        for field, value in updates.items():
            if field not in merged:
                continue
            if field in list_fields and value:
                # Keep rolling window: max 10 items, newest at end
                merged[field] = self._dedupe_list(merged[field], value)[-10:]
            elif field in scalar_fields and value is not None:
                merged[field] = value

        result = ContextMemory(**merged)
        self._write_raw(NS_CONTEXT, user_id, result.model_dump())
        return result

    # ------------------------------------------------------------------
    # Preference Memory
    # ------------------------------------------------------------------

    def get_preferences(self, user_id: str) -> PreferenceMemory:
        raw = self._read_raw(NS_PREFERENCE, user_id)
        if raw is None:
            return PreferenceMemory()
        return PreferenceMemory(**raw)

    def update_preferences(self, user_id: str, updates: Dict[str, Any]) -> PreferenceMemory:
        current = self.get_preferences(user_id)
        merged = current.model_dump()

        list_fields = {"sensitive_topics"}
        scalar_fields = {"conversation_style", "depth_preference", "response_length_preference"}
        bool_fields = {"prefers_short_questions"}

        for field, value in updates.items():
            if field not in merged:
                continue
            if field in list_fields and value:
                merged[field] = self._dedupe_list(merged[field], value)
            elif field in scalar_fields and value is not None:
                merged[field] = value
            elif field in bool_fields and value is not None:
                merged[field] = value

        result = PreferenceMemory(**merged)
        self._write_raw(NS_PREFERENCE, user_id, result.model_dump())
        return result

    # ------------------------------------------------------------------
    # Match Preference Memory
    # ------------------------------------------------------------------

    def get_match_preferences(self, user_id: str) -> MatchPreferenceMemory:
        raw = self._read_raw(NS_MATCH_PREFERENCE, user_id)
        if raw is None:
            return MatchPreferenceMemory()
        return MatchPreferenceMemory(**raw)

    def update_match_preferences(self, user_id: str, updates: Dict[str, Any]) -> MatchPreferenceMemory:
        current = self.get_match_preferences(user_id)
        merged = current.model_dump()

        for field, value in updates.items():
            if field not in merged or value is None:
                continue
            merged[field] = value

        result = MatchPreferenceMemory(**merged)
        if (
            result.edad_minima is not None
            and result.edad_maxima is not None
            and result.edad_minima > result.edad_maxima
        ):
            raise ValueError("edad_minima cannot be greater than edad_maxima.")

        self._write_raw(NS_MATCH_PREFERENCE, user_id, result.model_dump())
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_all(self, user_id: str) -> Dict[str, Any]:
        """Return all three memory blobs as plain dicts."""
        return {
            "profile_memory": self.get_profile(user_id).model_dump(),
            "context_memory": self.get_context(user_id).model_dump(),
            "preference_memory": self.get_preferences(user_id).model_dump(),
            "match_preference_memory": self.get_match_preferences(user_id).model_dump(),
        }

    def get_match_payload(self, user_id: str) -> Dict[str, Any]:
        """Return the exact payload expected by the external match service."""
        return {
            "user_id": user_id,
            "profile_memory": self.get_profile(user_id).model_dump(),
            "preference_memory": self.get_match_preferences(user_id).model_dump(),
            "profile_completion": self.compute_match_completion(user_id),
        }

    def compute_match_completion(self, user_id: str) -> float:
        """
        Completion score for the external match payload only.
        Conversation context and agent-only preferences do not affect this.
        """
        profile = self.get_profile(user_id)
        match_prefs = self.get_match_preferences(user_id)

        score = 0.0
        total = 16.0

        if profile.edad is not None:
            score += 1.0
        if profile.genero:
            score += 1.0
        if profile.bio:
            score += 1.0
        if profile.location:
            score += 1.0
        if len(profile.interests) >= 3:
            score += 2.0
        elif profile.interests:
            score += 1.0
        if len(profile.personality_traits) >= 2:
            score += 2.0
        elif profile.personality_traits:
            score += 1.0
        if profile.social_style:
            score += 1.0
        if profile.vibe_summary:
            score += 1.0
        if profile.hobbies:
            score += 1.0
        if profile.dislikes:
            score += 1.0
        if profile.favorite_environments:
            score += 1.0
        if profile.emotional_style:
            score += 1.0
        if match_prefs.edad_minima is not None and match_prefs.edad_maxima is not None:
            score += 1.0
        if match_prefs.distancia_maxima_km is not None:
            score += 0.5
        if match_prefs.genero_preferido:
            score += 0.5

        return round(min(score / total, 1.0), 2)

    def compute_completion(self, user_id: str) -> float:
        """
        Heuristic profile-completion score [0–1].
        Counts how many of the 'important' fields are populated.
        """
        profile = self.get_profile(user_id)
        prefs = self.get_preferences(user_id)
        context = self.get_context(user_id)

        score = 0.0
        total = 13.0  # max denominator

        if len(profile.interests) >= 3:
            score += 2.0
        elif profile.interests:
            score += 1.0

        if len(profile.personality_traits) >= 2:
            score += 2.0
        elif profile.personality_traits:
            score += 1.0

        if profile.social_style:
            score += 1.5
        if profile.vibe_summary:
            score += 1.5
        if profile.hobbies:
            score += 1.0
        if profile.dislikes:
            score += 1.0
        if profile.favorite_environments:
            score += 0.5
        if prefs.conversation_style:
            score += 0.5
        if context.recent_topics:
            score += 0.5
        if profile.emotional_style:
            score += 0.5
        if context.recent_life_changes:
            score += 0.5
        if context.current_mood_theme:
            score += 0.5
        if prefs.sensitive_topics:
            score += 0.5

        return round(min(score / total, 1.0), 2)


# ---------------------------------------------------------------------------
# Singleton — shared across the FastAPI app lifetime
# ---------------------------------------------------------------------------
def _build_store() -> MemoryStore:
    mongo_uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if mongo_uri:
        return MongoMemoryStore(
            uri=mongo_uri,
            database_name=os.getenv("MONGODB_DB_NAME", "allora_agent"),
            collection_name=os.getenv("MONGODB_COLLECTION", "memory"),
            server_selection_timeout_ms=int(os.getenv("MONGODB_TIMEOUT_MS", "5000")),
        )
    return InMemoryStore()


_store = _build_store()
memory_manager = MemoryManager(store=_store)

# Expose the raw store so LangGraph's checkpointer can share it
langgraph_store = _store
