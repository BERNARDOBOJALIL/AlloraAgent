"""
console_chat.py
────────────────
Cliente interactivo de consola para el agente de perfil de Allora.

Run with:
    python docs/console_chat.py

Server must be running first:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict

import httpx

BASE_URL = "http://localhost:8000"


def prompt_list(question: str, count: int) -> list[str]:
    while True:
        raw = input(f"{question} (puedes separarlas con comas): ").strip()
        if raw.lower() in {"skip", "saltar", "salta"}:
            return []

        items = [item.strip() for item in raw.split(",") if item.strip()]
        if len(items) >= count:
            return items[:count]

        print(f"Escribe al menos {count} opciones, separadas por comas.")


def print_help() -> None:
    print("\nComandos:")
    print("  /help               Muestra los comandos disponibles")
    print("  /profile            Muestra el perfil completo del usuario actual")
    print("  /thread [thread_id] Cambia a un hilo nuevo o específico")
    print("  /reset              Borra la memoria completa del usuario actual")
    print("  /exit               Sale del chat")


def print_onboarding_summary(hobbies: list[str], dislikes: list[str]) -> None:
    print("\nResumen inicial:")
    print(f"  Hobbies: {', '.join(hobbies) if hobbies else 'ninguno'}")
    print(f"  No le gusta: {', '.join(dislikes) if dislikes else 'nada indicado'}")


def print_agent_reply(data: Dict[str, Any]) -> None:
    print(f"\nALLORA: {data['assistant_message']}")

    state = data.get("conversation_state", {})
    completion = state.get("profile_completion", 0.0)
    should_continue = state.get("should_continue", True)

    print(f"Progress: {completion * 100:.0f}%")
    if not should_continue:
        print("(Agent marked conversation as complete for now)")


async def wait_for_server(client: httpx.AsyncClient) -> bool:
    for attempt in range(1, 6):
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5)
            resp.raise_for_status()
            return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            if attempt == 5:
                return False
            await asyncio.sleep(1)
    return False


async def fetch_profile(client: httpx.AsyncClient, user_id: str) -> None:
    resp = await client.get(f"{BASE_URL}/profile/{user_id}", timeout=15)
    resp.raise_for_status()
    profile = resp.json()
    print("\n=== PERFIL COMPLETO ===")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


async def reset_profile(client: httpx.AsyncClient, user_id: str) -> None:
    resp = await client.delete(f"{BASE_URL}/profile/{user_id}", timeout=15)
    resp.raise_for_status()
    print(f"\n{resp.json().get('message', 'Perfil eliminado.')}")


async def send_chat(
    client: httpx.AsyncClient,
    user_id: str,
    thread_id: str,
    message: str,
) -> Dict[str, Any]:
    payload = {"user_id": user_id, "thread_id": thread_id, "message": message}
    resp = await client.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


async def seed_onboarding(
    client: httpx.AsyncClient,
    user_id: str,
    thread_id: str,
    hobbies: list[str],
    dislikes: list[str],
) -> Dict[str, Any]:
    onboarding_lines = ["Respuestas iniciales:"]
    if hobbies:
        onboarding_lines.append(f"- Hobbies: {', '.join(hobbies)}")
    if dislikes:
        onboarding_lines.append(f"- Cosas que no le gustan / turn-offs: {', '.join(dislikes)}")
    onboarding_lines.append(
        "Usa esto como base del perfil y sigue con la mejor siguiente pregunta para descubrir más del usuario en contexto de citas."
    )
    return await send_chat(client, user_id, thread_id, "\n".join(onboarding_lines))


async def main() -> None:
    print("=" * 60)
    print("Chat de consola de Allora")
    print("=" * 60)

    user_id = input("ID de usuario (por defecto: alex_2025): ").strip() or "alex_2025"
    thread_id = f"session-{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient() as client:
        is_ready = await wait_for_server(client)
        if not is_ready:
            print("\nEl servidor no está disponible. Inícialo con:")
            print("  uvicorn app.main:app --reload")
            return

        print(f"\nConectado a {BASE_URL}")
        print(f"Usando user_id={user_id}")
        print(f"Usando thread_id={thread_id}")
        print_help()

        print("\nPara empezar, dame un poco de contexto para que yo lleve mejor la conversación.")
        hobbies = prompt_list("Dime 3 hobbies", 3)
        dislikes = prompt_list("Dime 3 cosas que no te gusten", 3)
        print_onboarding_summary(hobbies, dislikes)

        try:
            seed_data = await seed_onboarding(client, user_id, thread_id, hobbies, dislikes)
            print_agent_reply(seed_data)
        except httpx.HTTPStatusError as exc:
            print(f"\nError HTTP durante el inicio: {exc}")
        except httpx.ReadTimeout:
            print("\nEl inicio tardó demasiado. Sigo en modo chat de todas formas.")
        except httpx.HTTPError as exc:
            print(f"\nError de red durante el inicio: {exc}")

        while True:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nHasta luego.")
                return

            if not text:
                continue

            if text.lower() == "/exit":
                print("\nHasta luego.")
                return

            if text.lower() == "/help":
                print_help()
                continue

            if text.lower() == "/profile":
                try:
                    await fetch_profile(client, user_id)
                except httpx.HTTPError as exc:
                    print(f"\nError al obtener el perfil: {exc}")
                continue

            if text.lower() == "/reset":
                try:
                    await reset_profile(client, user_id)
                except httpx.HTTPError as exc:
                    print(f"\nError al borrar el perfil: {exc}")
                continue

            if text.lower().startswith("/thread"):
                parts = text.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    thread_id = parts[1].strip()
                else:
                    thread_id = f"session-{uuid.uuid4().hex[:8]}"
                print(f"\nNuevo thread_id={thread_id}")
                continue

            try:
                data = await send_chat(client, user_id, thread_id, text)
                print_agent_reply(data)
            except httpx.HTTPStatusError as exc:
                print(f"\nError HTTP: {exc}")
            except httpx.ReadTimeout:
                print("\nLa petición tardó demasiado. El servidor está ocupado o no responde.")
            except httpx.HTTPError as exc:
                print(f"\nError de red: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
