"""
agent.py — Allora profile-building agent powered by LangGraph.

Graph architecture
──────────────────
                    ┌──────────────────────┐
          START ──► │   load_memories      │
                    └─────────┬────────────┘
                              │
                    ┌─────────▼────────────┐
                    │   profile_agent      │  ◄── main LLM node
                    └─────────┬────────────┘
                              │
                    ┌─────────▼────────────┐
                    │  extract_and_save    │  ◄── Trustcall / structured extraction
                    └─────────┬────────────┘
                              │
                             END

Node responsibilities
─────────────────────
• load_memories     — Reads all 3 memory stores and injects them into the LLM context.
• profile_agent     — Generates a warm, natural reply AND a structured JSON payload
                      with memory_updates + conversation_state.
• extract_and_save  — Validates the structured payload, merges it into the stores,
                      and produces the final ChatResponse.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import os
import logging
import unicodedata

class _UnavailableModel:
    # Placeholder adapter with same `invoke(messages)` shape. It will raise
    # when called so `profile_agent`'s try/except will produce a friendly
    # offline JSON response instead of crashing the API.
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model", "llama-3.1-8b-instant")

    def invoke(self, messages):
        raise RuntimeError(
            "Groq model not available in this environment. "
            "Install `langchain-groq` and set GROQ_API_KEY in your .env.`"
        )


# Prefer Groq if available; fall back to a lightweight adapter that will
# raise on invoke so the existing fallback in `profile_agent` will handle it.
try:
    # langchain adapter for Groq may not be installed in this env. Try the
    # direct langchain-groq package first, then fallback to a generic import
    # if available. If neither is present, keep a placeholder that fails
    # predictably so we return the demo fallback instead of 500.
    from langchain_groq import ChatGroq  # type: ignore

    _model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
    )
except Exception:
    _model = _UnavailableModel()

# Diagnostic flag and logging at import time so startup logs show why we
# fall back to offline mode (helpful for users who set .env but haven't
# installed the Groq adapter).
_MODEL_READY = not isinstance(_model, _UnavailableModel)
if _MODEL_READY:
    logging.info("Groq model adapter initialized: %s", getattr(_model, "model_name", "<unknown>"))
else:
    logging.warning(
        "Groq model adapter NOT available. Install the adapter and set GROQ_API_KEY. "
        "Current GROQ_API_KEY present=%s",
        bool(os.environ.get("GROQ_API_KEY")),
    )
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from app.memory.memory_manager import memory_manager
from app.schemas.api import (
    ChatResponse,
    ConversationState,
    ContextMemoryUpdate,
    MemoryUpdates,
    PreferenceMemoryUpdate,
    ProfileMemoryUpdate,
)


def _latest_user_message(messages: List[Any]) -> str:
    """Return the latest human message content from the graph state."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "")
    return ""


def _latest_assistant_message_before_user(messages: List[Any]) -> str:
    """Return the assistant message immediately before the latest human turn."""
    seen_latest_user = False
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if not seen_latest_user:
                seen_latest_user = True
            continue
        if seen_latest_user and isinstance(message, AIMessage):
            return str(message.content or "")
    return ""


def _normalize_memory_item(item: str) -> str:
    item = item.strip().lower()
    item = unicodedata.normalize("NFKD", item)
    item = "".join(ch for ch in item if not unicodedata.combining(ch))
    item = re.sub(r"^[\W_]+|[\W_]+$", "", item)
    item = re.sub(
        r"^(a |al |la |el |los |las |en |de |del |hacer |ir a |to |the |going to )",
        "",
        item,
    )
    item = re.sub(r"\b(lol|jaja|haha|jeje)\b", "", item).strip()
    syn_map = {
        "horseback riding": "montar a caballo",
        "caballo": "montar a caballo",
        "natacion": "nadar",
        "swimming": "nadar",
        "cine": "ir al cine",
        "peliculas": "ir al cine",
        "movies": "ir al cine",
    }
    return syn_map.get(item, item)


def _split_memory_items(raw_items: str) -> list[str]:
    raw_items = re.sub(r"\b(?:pero|but)\b.*$", "", raw_items, flags=re.I).strip()
    parts = re.split(r",|;| y | e | and | & |/|\balso\b|\bademas\b", raw_items, flags=re.I)
    items: list[str] = []
    skipped_starts = (
        "prefiero ",
        "i prefer ",
        "no me ",
        "i don't ",
        "i do not ",
        "odio ",
        "detesto ",
    )
    value_items = {
        "sinceridad",
        "honestidad",
        "autenticidad",
        "confianza",
        "no juzgar",
        "no juicio",
    }
    for part in parts:
        raw_part = part.strip().lower()
        if not raw_part or raw_part.startswith(skipped_starts):
            continue
        normalized = _normalize_memory_item(part)
        if normalized in value_items:
            continue
        if normalized:
            items.append(normalized)
    return items


def _append_new(target: Dict[str, Any], field: str, incoming: list[str], existing: list[str]) -> None:
    seen = {str(item).lower() for item in existing}
    current = target.setdefault(field, [])
    for item in incoming:
        if item and item.lower() not in seen and item.lower() not in {x.lower() for x in current}:
            current.append(item)


_ENVIRONMENT_TERMS = {
    "lugares intimos",
    "lugares tranquilos",
    "ambientes informales",
    "ambientes tranquilos",
    "entornos tranquilos",
    "poco ruido",
    "cafeterias",
    "cafes",
    "coffee shops",
    "playa",
    "montana",
    "museos",
    "parques",
}

_CONVERSATION_INTEREST_TERMS = {
    "platicas interminables",
    "conversaciones profundas",
    "hablar de temas profundos",
    "temas profundos",
    "filosofia",
    "filosofia",
    "temas serios",
}

_LISTENING_TERMS = {
    "escuchar a la otra persona",
    "mas escuchar a la otra persona",
    "escuchar",
}


def _is_environment_item(item: str) -> bool:
    normalized = _normalize_memory_item(item)
    return (
        normalized in _ENVIRONMENT_TERMS
        or re.search(r"\b(lugares?|ambientes?|entornos?)\b", normalized) is not None
        or re.search(r"\b(intimos?|tranquilos?|poco ruido|silenciosos?|calmados?)\b", normalized) is not None
    )


def _is_conversation_interest_item(item: str) -> bool:
    normalized = _normalize_memory_item(item)
    return (
        normalized in _CONVERSATION_INTEREST_TERMS
        or re.search(r"\b(platicas?|conversaciones?|temas?)\b", normalized) is not None
        or re.search(r"\b(profundos?|serios?|filosofia|filosoficos?)\b", normalized) is not None
    )


def _is_listening_item(item: str) -> bool:
    normalized = _normalize_memory_item(item)
    return normalized in _LISTENING_TERMS or re.search(r"\bescuchar\b", normalized) is not None


def _route_profile_items(
    profile: Dict[str, Any],
    context: Dict[str, Any],
    items: list[str],
    profile_existing: Dict[str, Any],
    context_existing: Dict[str, Any],
) -> None:
    hobbies: list[str] = []
    interests: list[str] = []
    environments: list[str] = []
    traits: list[str] = []

    for item in items:
        if _is_environment_item(item):
            environments.append(item)
            continue
        if _is_listening_item(item):
            traits.append("attentive listener")
            interests.append("deep conversation")
            continue
        if _is_conversation_interest_item(item):
            interests.append(item)
            continue
        hobbies.append(item)
        interests.append(item)

    _append_new(profile, "hobbies", hobbies, profile_existing.get("hobbies", []))
    _append_new(profile, "interests", interests, profile_existing.get("interests", []))
    _append_new(profile, "favorite_environments", environments, profile_existing.get("favorite_environments", []))
    _append_new(profile, "personality_traits", traits, profile_existing.get("personality_traits", []))
    _append_new(context, "recent_topics", hobbies + interests + environments + traits, context_existing.get("recent_topics", []))


def _extract_user_memory_updates(
    text: str,
    existing_memory: Dict[str, Any],
    previous_assistant_text: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic safety net for high-signal profile facts in the user's own words.
    The LLM may still add richer summaries, but explicit user facts should not vanish.
    """
    profile_existing = existing_memory.get("profile_memory", {})
    context_existing = existing_memory.get("context_memory", {})
    prefs_existing = existing_memory.get("preference_memory", {})

    profile: Dict[str, Any] = {
        "interests": [],
        "personality_traits": [],
        "favorite_environments": [],
        "hobbies": [],
        "dislikes": [],
    }
    context: Dict[str, Any] = {
        "recent_topics": [],
        "evolving_interests": [],
        "recent_life_changes": [],
    }
    prefs: Dict[str, Any] = {"sensitive_topics": []}
    normalized_text = _normalize_memory_item(text)
    normalized_previous = _normalize_memory_item(previous_assistant_text)

    def _mask(pattern: str, source: str) -> str:
        for match in list(re.finditer(pattern, source, flags=re.I)):
            source = source[: match.start()] + (" " * (match.end() - match.start())) + source[match.end():]
        return source

    dislike_patterns = [
        r"(?:no me gusta(?:n)?|odio|detesto|me disgusta(?:n)?|no soporto|me caga(?:n)?) (?P<items>[^.\n?]+)",
        r"(?:i don't like|i do not like|i hate|not a fan of|i dislike) (?P<items>[^.\n?]+)",
        r"(?:dislikes|turn-offs|turnoffs|cosas que no le gustan)\s*:\s*(?P<items>[^.\n]+)",
    ]
    like_search_text = text
    for pattern in dislike_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            _append_new(profile, "dislikes", _split_memory_items(match.group("items")), profile_existing.get("dislikes", []))
            like_search_text = (
                like_search_text[: match.start()]
                + (" " * (match.end() - match.start()))
                + like_search_text[match.end():]
            )

    value_like_patterns = [
        r"(?:me gusta|me encanta|valoro|busco|prefiero) (?:la )?(?:sinceridad|honestidad|autenticidad|confianza)",
        r"(?:que )?(?:la otra persona )?(?:no me juzgue|no juzgue)",
        r"(?:sin temor a ser juzgado|sin miedo a ser juzgado|sin ser juzgado)",
    ]
    for pattern in value_like_patterns:
        like_search_text = _mask(pattern, like_search_text)

    like_patterns = [
        r"(?:^|[\s,;])(?:me gusta(?:n)?|me encanta(?:n)?|amo|disfruto|me apasiona) (?P<items>[^.\n?]+)",
        r"(?:i like|i love|i enjoy|i'm into|i am into|i've been into|i have been into) (?P<items>[^.\n?]+)",
        r"(?:hobbies|pasatiempos)\s*:\s*(?P<items>[^.\n]+)",
        r"(?:mis hobbies son|my hobbies are) (?P<items>[^.\n?]+)",
    ]
    for pattern in like_patterns:
        for match in re.finditer(pattern, like_search_text, flags=re.I):
            items = _split_memory_items(match.group("items"))
            _route_profile_items(profile, context, items, profile_existing, context_existing)

    activity_patterns = [
        r"(?:en mi tiempo libre|suelo|usually|i usually|i spend .*? on|i started going to|empece a ir a|empecé a ir a) (?P<items>[^.\n?]+)",
    ]
    for pattern in activity_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            items = _split_memory_items(match.group("items"))
            _route_profile_items(profile, context, items, profile_existing, context_existing)

    environment_patterns = [
        r"(?:me siento mejor en|me encanta estar en|prefiero lugares como|i feel best in|i love being in|i prefer places like) (?P<items>[^.\n?]+)",
        r"(?:coffee shops|cafes|cafeterias|playa|beach|montana|mountains|rooftop bars|clubs|museums)",
    ]
    for pattern in environment_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value = match.groupdict().get("items") or match.group(0)
            _append_new(
                profile,
                "favorite_environments",
                _split_memory_items(value),
                profile_existing.get("favorite_environments", []),
            )

    trait_terms = {
        "introvert": "introvert-leaning",
        "introverted": "introvert-leaning",
        "introvertido": "introvertido",
        "introvertida": "introvertida",
        "extrovert": "extrovert-leaning",
        "extroverted": "extrovert-leaning",
        "extrovertido": "extrovertido",
        "extrovertida": "extrovertida",
        "creative": "creative",
        "creativo": "creativo",
        "creativa": "creativa",
        "curious": "curious",
        "curioso": "curioso",
        "curiosa": "curiosa",
        "spontaneous": "spontaneous",
        "espontaneo": "espontaneo",
        "espontanea": "espontanea",
    }
    found_traits = [trait for term, trait in trait_terms.items() if re.search(rf"\b{re.escape(term)}\b", normalized_text)]
    _append_new(profile, "personality_traits", found_traits, profile_existing.get("personality_traits", []))

    if re.search(r"\b(murio mi perro|murio mi mascota|perdi a mi perro|perdi a mi mascota|murio mi gato|perdi a mi gato)\b", normalized_text):
        _append_new(
            context,
            "recent_life_changes",
            ["recently lost a pet"],
            context_existing.get("recent_life_changes", []),
        )
        context["current_mood_theme"] = "grieving the recent loss of a pet"
        _append_new(prefs, "sensitive_topics", ["pet loss"], prefs_existing.get("sensitive_topics", []))

    if re.search(r"\b(personas en las que confio|personas que confio|people i trust|personas de confianza)\b", normalized_text):
        profile["emotional_style"] = "comfortable sharing feelings with people they trust"
        _append_new(
            profile,
            "personality_traits",
            ["trust-oriented", "emotionally open with trusted people"],
            profile_existing.get("personality_traits", []),
        )

    if re.search(r"\b(no me juzgue|no juzgue|no me juzgan|being judged|sin ser juzgado)\b", normalized_text):
        _append_new(profile, "dislikes", ["being judged"], profile_existing.get("dislikes", []))
        _append_new(
            profile,
            "personality_traits",
            ["values nonjudgmental connection"],
            profile_existing.get("personality_traits", []),
        )
        prefs["conversation_style"] = "warm, accepting, and nonjudgmental"

    if re.search(r"\b(sinceridad|honestidad|autenticidad)\b", normalized_text):
        _append_new(profile, "personality_traits", ["values sincerity"], profile_existing.get("personality_traits", []))
        prefs["conversation_style"] = "honest and direct"

    if re.search(r"\b(no me siento comodo en conflictos|no estoy comodo en conflictos|evito conflictos|conflictos)\b", normalized_text):
        _append_new(profile, "dislikes", ["conflict"], profile_existing.get("dislikes", []))
        _append_new(profile, "personality_traits", ["conflict-sensitive"], profile_existing.get("personality_traits", []))
        _append_new(prefs, "sensitive_topics", ["conflict"], prefs_existing.get("sensitive_topics", []))
        profile["emotional_style"] = "sensitive around conflict and prefers emotionally safe conversations"

    if re.search(r"\b(lloro mucho|llora mucho|llorar mucho|i cry a lot)\b", normalized_text):
        _append_new(profile, "personality_traits", ["emotionally expressive"], profile_existing.get("personality_traits", []))
        profile["emotional_style"] = "emotionally expressive and can cry easily, especially around conflict"

    yes_to_context = re.fullmatch(r"(si|sí|yes|claro|totalmente|me encantaria|me encantaria mucho|si me encantaria|si, me encantaria)[\s.!?]*", normalized_text)
    if yes_to_context and re.search(r"\b(sin temor a ser juzgado|sin miedo a ser juzgado|ser completamente tu mismo|completamente tu mismo)\b", normalized_previous):
        _append_new(
            profile,
            "personality_traits",
            ["wants to feel fully accepted"],
            profile_existing.get("personality_traits", []),
        )
        _append_new(profile, "dislikes", ["feeling judged"], profile_existing.get("dislikes", []))
        prefs["conversation_style"] = "accepting and emotionally safe"
    if yes_to_context and re.search(r"\b(directa y sincera|directo y sincero|cosas dificiles|sincera contigo|sincero contigo)\b", normalized_previous):
        _append_new(profile, "personality_traits", ["values direct honesty"], profile_existing.get("personality_traits", []))
        prefs["conversation_style"] = "honest and direct"

    if re.search(r"\b(small groups?|grupos pequenos|planes tranquilos|quiet plans)\b", normalized_text):
        profile["social_style"] = "prefers quieter plans or small-group settings"
    elif re.search(r"\b(group stuff|grupos|parties|fiestas|salir mas|saying yes more)\b", normalized_text):
        profile["social_style"] = "open to group plans and saying yes socially"

    if profile["favorite_environments"] and not profile.get("social_style"):
        profile["social_style"] = "prefers calm, intimate, low-noise environments"
    if any(_is_conversation_interest_item(item) for item in profile["interests"]) or any(
        trait == "attentive listener" for trait in profile["personality_traits"]
    ):
        _append_new(
            profile,
            "personality_traits",
            ["reflective", "drawn to deep conversation"],
            profile_existing.get("personality_traits", []),
        )
    if (
        (profile["favorite_environments"] or profile["personality_traits"] or profile["interests"])
        and not profile_existing.get("vibe_summary")
    ):
        vibe_parts = []
        if profile["favorite_environments"]:
            vibe_parts.append("prefers calm, intimate spaces")
        if any(_is_conversation_interest_item(item) for item in profile["interests"]):
            vibe_parts.append("enjoys deep, thoughtful conversations")
        if "attentive listener" in profile["personality_traits"]:
            vibe_parts.append("likes listening closely to others")
        if vibe_parts:
            profile["vibe_summary"] = "Reflective person who " + ", ".join(vibe_parts) + "."

    if re.search(r"\b(overthink|sobrepienso|me cuesta abrirme|open up|me abro|process internally)\b", normalized_text):
        profile["emotional_style"] = "processes feelings internally at first, then opens up with trust"

    if re.search(r"\b(short questions|preguntas cortas|breve|directo|directa)\b", normalized_text):
        prefs["prefers_short_questions"] = True
        if not prefs_existing.get("conversation_style"):
            prefs["conversation_style"] = "direct and concise"

    if re.search(r"\b(rough week|semana dificil|semana dura|feeling better|me siento mejor)\b", normalized_text):
        context["current_mood_theme"] = "recovering from a rough week but feeling better"

    life_patterns = [
        r"(?:just moved|me mude|me mudé|new city|nueva ciudad|new job|nuevo trabajo|started a new|empece un nuevo|empecé un nuevo)(?P<rest>[^.\n?]*)",
    ]
    for pattern in life_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            update = _normalize_memory_item((match.group(0) or "").strip())
            _append_new(context, "recent_life_changes", [update], context_existing.get("recent_life_changes", []))

    if profile["hobbies"] or profile["interests"] or profile["dislikes"] or profile["personality_traits"]:
        topics = profile["hobbies"] + profile["interests"] + profile["dislikes"] + profile["personality_traits"]
        _append_new(context, "recent_topics", topics, context_existing.get("recent_topics", []))

    return {"profile": profile, "context": context, "preferences": prefs}


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AlloraState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_id: str
    thread_id: str
    # Injected by load_memories, consumed by profile_agent
    memory_context: str
    # Filled by profile_agent, consumed by extract_and_save
    raw_agent_output: str
    # Final result passed back to FastAPI
    chat_response: Optional[ChatResponse]


# ---------------------------------------------------------------------------
# System prompt factory
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """
You are Allora's profile-building companion — warm, curious, and genuinely interested in who this person is.

Your ONLY job is to learn about the user's:
• personality and vibe
• interests and hobbies
• social style and behavior
• emotional preferences
• conversational style

You do NOT ask about age, gender, relationship goals, or profile settings. Those are handled elsewhere.

Guidelines
──────────
- Sound like a friend, not an interviewer. Never ask more than ONE question per turn.
- React naturally to what they say before pivoting to something new.
- Follow threads briefly, but do not get stuck on a single topic for many turns.
- Be playful, witty, empathetic. Mirror their energy.
- If they seem to want to stop, respect that immediately.
- Never invent or assume information they haven't shared.
- Until the profile is complete, ALWAYS end with exactly ONE follow-up question.
- Actively ask about dislikes / turn-offs (what they do not enjoy) because it is critical for dating compatibility.

Conversation steering policy (important):
- Your mission is profile coverage, not long topic chat.
- In each turn, do this sequence:
    1) brief acknowledgment of user's message,
    2) optional micro-follow-up tied to what they just said,
    3) one question that unlocks a missing profile area.
- Prioritize missing areas listed in memory context.
- Do not ask more than 2 consecutive questions about the same category.
- Prefer broadening profile coverage over deep diving one hobby unless user clearly insists.

Current memory snapshot (DO NOT repeat this back verbatim):
{memory_context}

Response format (CRITICAL — you MUST return ONLY valid JSON, nothing else):
{{
  "assistant_message": "<your warm, natural reply here>",
  "memory_updates": {{
    "profile_memory": {{
      "interests": [],
      "traits": [],
      "social_style": null,
      "vibe_summary": null,
      "favorite_environments": [],
      "hobbies": [],
            "dislikes": [],
      "emotional_style": null
    }},
    "context_memory": {{
      "recent_topics": [],
      "evolving_interests": [],
      "life_updates": [],
      "recent_social_behavior": null,
      "current_mood_theme": null
    }},
    "preference_memory": {{
      "conversation_style": null,
      "prefers_short_questions": false,
      "depth_preference": null,
      "sensitive_topics": []
    }}
  }},
  "conversation_state": {{
    "profile_completion": {profile_completion},
    "should_continue": true
  }}
}}

Rules for memory_updates:
- Only populate fields you learned something NEW about in this turn.
- Treat every user answer as profile signal. If the user shares a feeling, value, boundary,
  fear, grief, conflict style, trust condition, or communication preference, save it.
- Values like sincerity, honesty, trust, authenticity, and non-judgment belong in traits,
  emotional_style, dislikes, or conversation_style. Do not save them as hobbies.
- Losses, difficult weeks, moves, job changes, and other life events belong in context_memory.
- Lists should contain ONLY new items (not duplicates of what's already stored).
- Scalars should be null if unchanged.
- profile_completion is a float between 0.0 and 1.0 based on how complete you feel the profile is.
- Set should_continue to true while profile is incomplete; set false only if user explicitly wants to stop or profile is complete.
""".strip()


# Model is defined above with a best-effort Groq client or a placeholder.


# ---------------------------------------------------------------------------
# Node: load_memories
# ---------------------------------------------------------------------------

def load_memories(state: AlloraState) -> Dict[str, Any]:
    """
    Reads all three memory stores and formats them as a concise
    context block injected into the system prompt.
    """
    user_id = state["user_id"]
    all_mem = memory_manager.get_all(user_id)
    completion = memory_manager.compute_completion(user_id)

    profile = all_mem["profile_memory"]
    context = all_mem["context_memory"]
    prefs = all_mem["preference_memory"]

    lines = ["=== What I already know about this user ==="]

    # Profile
    if any([profile["interests"], profile["personality_traits"],
            profile["social_style"], profile["vibe_summary"],
            profile["hobbies"], profile["dislikes"]]):
        lines.append("\n[Profile]")
        if profile["interests"]:
            lines.append(f"  Interests: {', '.join(profile['interests'])}")
        if profile["personality_traits"]:
            lines.append(f"  Traits: {', '.join(profile['personality_traits'])}")
        if profile["hobbies"]:
            lines.append(f"  Hobbies: {', '.join(profile['hobbies'])}")
        if profile["dislikes"]:
            lines.append(f"  Dislikes / turn-offs: {', '.join(profile['dislikes'])}")
        if profile["favorite_environments"]:
            lines.append(f"  Favorite environments: {', '.join(profile['favorite_environments'])}")
        if profile["social_style"]:
            lines.append(f"  Social style: {profile['social_style']}")
        if profile["vibe_summary"]:
            lines.append(f"  Vibe: {profile['vibe_summary']}")
        if profile["emotional_style"]:
            lines.append(f"  Emotional style: {profile['emotional_style']}")
    else:
        lines.append("\n[Profile] Nothing yet — fresh conversation.")

    # Context
    if any([context["recent_topics"], context["recent_life_changes"],
            context["recent_social_behavior"]]):
        lines.append("\n[Recent Context]")
        if context["recent_topics"]:
            lines.append(f"  Recent topics: {', '.join(context['recent_topics'])}")
        if context["recent_life_changes"]:
            lines.append(f"  Life updates: {', '.join(context['recent_life_changes'])}")
        if context["recent_social_behavior"]:
            lines.append(f"  Social behavior lately: {context['recent_social_behavior']}")
        if context["current_mood_theme"]:
            lines.append(f"  Mood: {context['current_mood_theme']}")

    # Preferences
    if prefs["conversation_style"] or prefs["prefers_short_questions"]:
        lines.append("\n[Conversation Preferences]")
        if prefs["conversation_style"]:
            lines.append(f"  Style: {prefs['conversation_style']}")
        if prefs["prefers_short_questions"]:
            lines.append("  Prefers: short, focused questions")
        if prefs["depth_preference"]:
            lines.append(f"  Depth: {prefs['depth_preference']}")

    missing_areas: List[str] = []
    if not profile["interests"]:
        missing_areas.append("interests")
    if not profile["hobbies"]:
        missing_areas.append("hobbies")
    if not profile["dislikes"]:
        missing_areas.append("dislikes / turn-offs")
    if not profile["personality_traits"]:
        missing_areas.append("personality traits")
    if not profile["social_style"]:
        missing_areas.append("social style")
    if not profile["emotional_style"]:
        missing_areas.append("emotional style")
    if not profile["favorite_environments"]:
        missing_areas.append("favorite environments")
    if not prefs["conversation_style"]:
        missing_areas.append("conversation style preference")
    if not prefs["depth_preference"]:
        missing_areas.append("depth preference")

    if missing_areas:
        lines.append("\n[Missing Profile Areas — prioritize these]")
        lines.append(f"  {', '.join(missing_areas[:6])}")

    lines.append(f"\n[Completion] ~{int(completion * 100)}% of profile filled")
    lines.append("===========================================")

    return {
        "memory_context": "\n".join(lines),
        "chat_response": None,
    }


# ---------------------------------------------------------------------------
# Node: profile_agent
# ---------------------------------------------------------------------------

def profile_agent(state: AlloraState) -> Dict[str, Any]:
    """
    Main LLM node. Calls the Groq chat model with the memory-enriched system prompt
    and the full conversation history. Returns raw JSON string.
    """
    user_id = state["user_id"]
    completion = memory_manager.compute_completion(user_id)

    system_prompt = _SYSTEM_TEMPLATE.format(
        memory_context=state.get("memory_context", "No prior context."),
        profile_completion=completion,
    )

    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]

    try:
        response = _model.invoke(messages_to_send)
        if isinstance(response.content, str):
            raw_text = response.content
        elif isinstance(response.content, list):
            # Some providers return a list of content blocks.
            chunks: List[str] = []
            for block in response.content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
                elif isinstance(block, str) and block.strip():
                    chunks.append(block)
            raw_text = "\n".join(chunks).strip()
        else:
            raw_text = str(response.content)
    except Exception as exc:  # pragma: no cover - runtime fallback for missing API / runtime errors
        # Graceful fallback: return a minimal, valid JSON payload so the
        # extract_and_save node can continue and the API doesn't return 500.
        import logging

        logging.exception("LLM invocation failed; using offline fallback response")

        fallback = {
            "assistant_message": (
                "Lo siento — el modelo no está disponible ahora mismo. "
                "Estoy en modo demo local, intenta configurar la clave de API."
            ),
            "memory_updates": {
                "profile_memory": {
                    "interests": [],
                    "traits": [],
                    "social_style": None,
                    "vibe_summary": None,
                    "favorite_environments": [],
                    "hobbies": [],
                    "dislikes": [],
                    "emotional_style": None,
                },
                "context_memory": {
                    "recent_topics": [],
                    "evolving_interests": [],
                    "life_updates": [],
                    "recent_social_behavior": None,
                    "current_mood_theme": None,
                },
                "preference_memory": {
                    "conversation_style": None,
                    "prefers_short_questions": False,
                    "depth_preference": None,
                    "sensitive_topics": [],
                },
            },
            "conversation_state": {"profile_completion": completion, "should_continue": True},
        }

        raw_text = json.dumps(fallback)

    return {"raw_agent_output": raw_text}


# ---------------------------------------------------------------------------
# Node: extract_and_save
# ---------------------------------------------------------------------------

def _safe_parse_json(raw: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from the model output, even if wrapped in
    markdown code fences or prefixed with stray text.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    # Find first '{' to last '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model output.")
    return json.loads(cleaned[start:end])


def _user_wants_to_stop(messages: List[Any]) -> bool:
    """Heuristic stop-intent detector from the latest user message."""
    last_user = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user = (m.content or "").strip().lower()
            break

    if not last_user:
        return False

    stop_patterns = [
        "that's enough",
        "that is enough",
        "enough for now",
        "stop",
        "bye",
        "adios",
        "me voy",
        "ya fue",
        "gracias, eso es todo",
        "ya es todo",
    ]
    return any(p in last_user for p in stop_patterns)


def extract_and_save(state: AlloraState) -> Dict[str, Any]:
    """
    Parses the agent's JSON output, validates it with Pydantic,
    persists memory updates, and builds the final ChatResponse.
    """
    user_id = state["user_id"]
    raw = state.get("raw_agent_output", "")
    existing_memory = memory_manager.get_all(user_id)
    latest_user_text = _latest_user_message(state["messages"])
    previous_assistant_text = _latest_assistant_message_before_user(state["messages"])
    user_extracted = _extract_user_memory_updates(
        latest_user_text,
        existing_memory,
        previous_assistant_text=previous_assistant_text,
    )

    def _merge_extracted_updates(
        profile_delta: Dict[str, Any],
        context_delta: Dict[str, Any],
        prefs_delta: Dict[str, Any],
    ) -> None:
        extracted_profile = user_extracted["profile"]
        extracted_context = user_extracted["context"]
        extracted_prefs = user_extracted["preferences"]

        list_profile_fields = {
            "interests",
            "personality_traits",
            "favorite_environments",
            "hobbies",
            "dislikes",
        }
        for field in list_profile_fields:
            if extracted_profile.get(field):
                current = profile_delta.setdefault(field, [])
                seen = {str(item).lower() for item in current}
                for item in extracted_profile[field]:
                    if item.lower() not in seen:
                        current.append(item)
                        seen.add(item.lower())

        for field in ("social_style", "vibe_summary", "emotional_style"):
            if extracted_profile.get(field) and not profile_delta.get(field):
                profile_delta[field] = extracted_profile[field]

        for field in ("recent_topics", "evolving_interests", "recent_life_changes"):
            if extracted_context.get(field):
                current = context_delta.setdefault(field, [])
                seen = {str(item).lower() for item in current}
                for item in extracted_context[field]:
                    if item.lower() not in seen:
                        current.append(item)
                        seen.add(item.lower())

        for field in ("recent_social_behavior", "current_mood_theme"):
            if extracted_context.get(field) and not context_delta.get(field):
                context_delta[field] = extracted_context[field]

        if extracted_prefs.get("sensitive_topics"):
            current = prefs_delta.setdefault("sensitive_topics", [])
            seen = {str(item).lower() for item in current}
            for item in extracted_prefs["sensitive_topics"]:
                if item.lower() not in seen:
                    current.append(item)
                    seen.add(item.lower())
        for field in ("conversation_style", "depth_preference"):
            if extracted_prefs.get(field) and not prefs_delta.get(field):
                prefs_delta[field] = extracted_prefs[field]
        if extracted_prefs.get("prefers_short_questions"):
            prefs_delta["prefers_short_questions"] = True

    try:
        payload = _safe_parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        cleaned_raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        profile_delta: Dict[str, Any] = {
            "interests": [],
            "personality_traits": [],
            "vibe_summary": None,
            "favorite_environments": [],
            "hobbies": [],
            "dislikes": [],
        }
        context_delta: Dict[str, Any] = {"recent_topics": []}
        prefs_delta: Dict[str, Any] = {}
        _merge_extracted_updates(profile_delta, context_delta, prefs_delta)

        if any(v for v in profile_delta.values()) or any(v for v in context_delta.values()) or any(v for v in prefs_delta.values()):
            memory_manager.update_profile(user_id, profile_delta)
            memory_manager.update_context(user_id, context_delta)
            memory_manager.update_preferences(user_id, prefs_delta)

            actual_completion = memory_manager.compute_completion(user_id)
            turn_count = len([m for m in state["messages"] if isinstance(m, HumanMessage)])

            return {
                "chat_response": ChatResponse(
                    assistant_message=cleaned_raw or "",
                    memory_updates=MemoryUpdates(
                        profile_memory=ProfileMemoryUpdate(
                            interests=profile_delta.get("interests", []),
                            traits=profile_delta.get("personality_traits", []),
                            social_style=profile_delta.get("social_style"),
                            vibe_summary=profile_delta.get("vibe_summary"),
                            favorite_environments=profile_delta.get("favorite_environments", []),
                            hobbies=profile_delta.get("hobbies", []),
                            dislikes=profile_delta.get("dislikes", []),
                            emotional_style=profile_delta.get("emotional_style"),
                        ),
                        context_memory=ContextMemoryUpdate(
                            recent_topics=context_delta.get("recent_topics", []),
                            evolving_interests=context_delta.get("evolving_interests", []),
                            life_updates=context_delta.get("recent_life_changes", []),
                            recent_social_behavior=context_delta.get("recent_social_behavior"),
                            current_mood_theme=context_delta.get("current_mood_theme"),
                        ),
                        preference_memory=PreferenceMemoryUpdate(
                            conversation_style=prefs_delta.get("conversation_style"),
                            prefers_short_questions=prefs_delta.get("prefers_short_questions", False),
                            depth_preference=prefs_delta.get("depth_preference"),
                            sensitive_topics=prefs_delta.get("sensitive_topics", []),
                        ),
                    ),
                    conversation_state=ConversationState(
                        profile_completion=actual_completion,
                        should_continue=actual_completion < 1.0,
                        turn_count=turn_count,
                    ),
                )
            }

        assistant_fallback = cleaned_raw or (
            "Hey! Something went a little sideways on my end. What were you saying? 😊"
        )

        return {
            "chat_response": ChatResponse(
                assistant_message=assistant_fallback,
                memory_updates=MemoryUpdates(),
                conversation_state=ConversationState(
                    profile_completion=memory_manager.compute_completion(user_id),
                    should_continue=True,
                    turn_count=len(state["messages"]),
                ),
            )
        }

    assistant_message = payload.get("assistant_message", "")
    raw_updates = payload.get("memory_updates", {})
    raw_profile = raw_updates.get("profile_memory", {})
    raw_context = raw_updates.get("context_memory", {})
    raw_prefs = raw_updates.get("preference_memory", {})

    profile_delta: Dict[str, Any] = {}
    context_delta: Dict[str, Any] = {}
    prefs_delta: Dict[str, Any] = {}

    if raw_profile:
        profile_delta = {
            "interests": raw_profile.get("interests") or [],
            "personality_traits": raw_profile.get("traits") or [],
            "social_style": raw_profile.get("social_style"),
            "vibe_summary": raw_profile.get("vibe_summary"),
            "favorite_environments": raw_profile.get("favorite_environments") or [],
            "hobbies": raw_profile.get("hobbies") or [],
            "dislikes": raw_profile.get("dislikes") or [],
            "emotional_style": raw_profile.get("emotional_style"),
        }

    if raw_context:
        context_delta = {
            "recent_topics": raw_context.get("recent_topics") or [],
            "evolving_interests": raw_context.get("evolving_interests") or [],
            "recent_life_changes": raw_context.get("life_updates") or [],
            "recent_social_behavior": raw_context.get("recent_social_behavior"),
            "current_mood_theme": raw_context.get("current_mood_theme"),
        }

    if raw_prefs:
        prefs_delta = {
            "conversation_style": raw_prefs.get("conversation_style"),
            "prefers_short_questions": raw_prefs.get("prefers_short_questions", False),
            "depth_preference": raw_prefs.get("depth_preference"),
            "sensitive_topics": raw_prefs.get("sensitive_topics") or [],
        }

    _merge_extracted_updates(profile_delta, context_delta, prefs_delta)

    if any(v for v in profile_delta.values()):
        memory_manager.update_profile(user_id, profile_delta)
    if any(v for v in context_delta.values()):
        memory_manager.update_context(user_id, context_delta)
    if any(v for v in prefs_delta.values()):
        memory_manager.update_preferences(user_id, prefs_delta)

    actual_completion = memory_manager.compute_completion(user_id)
    turn_count = len([m for m in state["messages"] if isinstance(m, HumanMessage)])
    user_stop_intent = _user_wants_to_stop(state["messages"])
    should_continue = (not user_stop_intent) and (actual_completion < 1.0)

    chat_response = ChatResponse(
        assistant_message=assistant_message,
        memory_updates=MemoryUpdates(
            profile_memory=ProfileMemoryUpdate(
                interests=profile_delta.get("interests", []),
                traits=profile_delta.get("personality_traits", []),
                social_style=profile_delta.get("social_style"),
                vibe_summary=profile_delta.get("vibe_summary"),
                favorite_environments=profile_delta.get("favorite_environments", []),
                hobbies=profile_delta.get("hobbies", []),
                dislikes=profile_delta.get("dislikes", []),
                emotional_style=profile_delta.get("emotional_style"),
            ),
            context_memory=ContextMemoryUpdate(
                recent_topics=context_delta.get("recent_topics", []),
                evolving_interests=context_delta.get("evolving_interests", []),
                life_updates=context_delta.get("recent_life_changes", []),
                recent_social_behavior=context_delta.get("recent_social_behavior"),
                current_mood_theme=context_delta.get("current_mood_theme"),
            ),
            preference_memory=PreferenceMemoryUpdate(
                conversation_style=prefs_delta.get("conversation_style"),
                prefers_short_questions=prefs_delta.get("prefers_short_questions", False),
                depth_preference=prefs_delta.get("depth_preference"),
                sensitive_topics=prefs_delta.get("sensitive_topics", []),
            ),
        ),
        conversation_state=ConversationState(
            profile_completion=actual_completion,
            should_continue=should_continue,
            turn_count=turn_count,
        ),
    )

    return {"chat_response": chat_response}


def build_graph() -> Any:
    builder = StateGraph(AlloraState)

    builder.add_node("load_memories", load_memories)
    builder.add_node("profile_agent", profile_agent)
    builder.add_node("extract_and_save", extract_and_save)

    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "profile_agent")
    builder.add_edge("profile_agent", "extract_and_save")
    builder.add_edge("extract_and_save", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
