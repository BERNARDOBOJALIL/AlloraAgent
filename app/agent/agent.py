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
                    │  extract_and_save    │  ◄── JSON validation + memory save
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

class _UnavailableModel:
    # Placeholder adapter with same `invoke(messages)` shape. It will raise
    # when called so `profile_agent`'s try/except will produce a friendly
    # offline JSON response instead of crashing the API.
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model", os.getenv("ALLORA_GROQ_MODEL", "llama-3.3-70b-versatile"))

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

    GROQ_MODEL = os.getenv("ALLORA_GROQ_MODEL", "llama-3.3-70b-versatile")
    _model = ChatGroq(
        model=GROQ_MODEL,
        temperature=float(os.getenv("ALLORA_MODEL_TEMPERATURE", "0.35")),
        max_tokens=int(os.getenv("ALLORA_MODEL_MAX_TOKENS", "2048")),
        model_kwargs={"response_format": {"type": "json_object"}},
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
from langchain_core.messages import HumanMessage, SystemMessage
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


PROFILE_LIST_FIELDS = {"interests", "personality_traits", "favorite_environments", "hobbies", "dislikes"}
PROFILE_SCALAR_FIELDS = {"social_style", "vibe_summary", "emotional_style"}
PROFILE_EDIT_FIELDS = PROFILE_LIST_FIELDS | PROFILE_SCALAR_FIELDS

PROFILE_FIELD_DESCRIPTIONS = {
    "interests": "topics, activities, tastes, or passions the user genuinely cares about",
    "personality_traits": "observable personality traits or values, phrased as concise human traits",
    "favorite_environments": "places, settings, or atmospheres where the user feels comfortable or alive",
    "hobbies": "specific recurring hobbies or activities the user does",
    "dislikes": "things the user dislikes, avoids, or considers turn-offs",
    "social_style": "one concise sentence about how the user tends to socialize",
    "vibe_summary": "one warm one-or-two sentence summary of the user's overall vibe",
    "emotional_style": "one concise sentence about how the user processes or expresses emotions",
}


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


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
            elif isinstance(block, str) and block.strip():
                chunks.append(block)
        return "\n".join(chunks).strip()
    return str(content)


def _dedupe_clean_items(items: List[Any]) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        value = re.sub(r"\s+", " ", item).strip(" \t\r\n-•.,;:")
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            cleaned.append(value)
            seen.add(key)
    return cleaned


async def format_profile_field_with_model(
    field: str,
    user_text: str,
    current_value: Any = None,
) -> str | List[str]:
    """
    Convert a free-form manual edit prompt into exactly one profile field value.
    The model may interpret and clean the user's text, but it must not update
    any field other than the requested one.
    """
    if field not in PROFILE_EDIT_FIELDS:
        raise ValueError(f"Unknown profile field: {field}")

    expected_type = "array of short strings" if field in PROFILE_LIST_FIELDS else "single concise string"
    system_prompt = f"""
You format manual edits for a dating profile memory system.

Requested field: {field}
Field meaning: {PROFILE_FIELD_DESCRIPTIONS[field]}
Expected value type: {expected_type}

Rules:
- Use ONLY the user's text and the requested field.
- Do not infer unrelated fields.
- Preserve the user's language.
- Remove filler such as "me gusta", "soy", "quiero cambiarlo a" when it is not part of the actual value.
- If the text is unusable for this field, return an empty array for list fields or null for scalar fields.
- Return ONLY valid JSON in this shape: {{"value": ...}}
""".strip()

    human_prompt = json.dumps(
        {
            "field": field,
            "current_value": current_value,
            "user_text": user_text,
        },
        ensure_ascii=False,
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    try:
        if hasattr(_model, "ainvoke"):
            response = await _model.ainvoke(messages)
        else:
            response = _model.invoke(messages)
    except Exception as exc:
        logging.exception("Profile field edit model invocation failed")
        raise RuntimeError("Profile edit model unavailable.") from exc

    payload = _safe_parse_json(_message_content_to_text(response.content))
    value = payload.get("value")

    if field in PROFILE_LIST_FIELDS:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Model returned a non-list value for a list profile field.")
        return _dedupe_clean_items(value)

    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Model returned a non-string value for a scalar profile field.")
    return re.sub(r"\s+", " ", value).strip()


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
