# Allora Profile Agent

Allora es un servicio que construye y mantiene un perfil social estructurado
para cada usuario a partir de conversaciones y ediciones manuales. El agente
usa un LLM para generar respuestas conversacionales y para formatear
ediciones de un solo campo del perfil cuando el usuario prefiere no chatear.

Este repositorio contiene la API y la lógica principal para:

- Interactuar con el agente conversacional (`POST /chat`).
- Consultar, editar y generar campos individuales del perfil.
- Persistir memoria de perfil, contexto y preferencias.

Archivos clave
- `app/main.py` — definición de endpoints FastAPI.
- `app/agent/agent.py` — lógica del agente y utilidades para formatear campos.
- `app/memory/memory_manager.py` — abstracción del almacenamiento de memoria.
- `app/schemas/api.py` — modelos Pydantic para peticiones y respuestas.

Requisitos
- Python 3.10+
- Dependencias en `requirements.txt`.
- Variables de entorno (opcional, necesarias para usar un proveedor LLM real):
  - `GROQ_API_KEY` — clave para el adaptador Groq si se usa.
  - `ALLORA_GROQ_MODEL` — nombre del modelo (opcional).
  - `ALLORA_MODEL_TEMPERATURE`, `ALLORA_MODEL_MAX_TOKENS` — parámetros del modelo.

Instalación y ejecución local

1. Crear un entorno y activar:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\Activate     # Windows (PowerShell)
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Crear un fichero `.env` si quiere configurar la API del modelo y otras variables.

4. Ejecutar la API:

```bash
uvicorn app.main:app --reload --port 8000
```

La documentación interactiva estará disponible en `http://localhost:8000/docs`.

Persistencia con MongoDB Atlas

Para que la memoria sobreviva reinicios del servidor, configure estas variables en `.env` o en el dashboard del deploy:

```bash
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/?retryWrites=true&w=majority
MONGODB_DB_NAME=allora_agent
MONGODB_COLLECTION=memory
```

`MONGODB_URI` tambien puede llamarse `MONGO_URI`. Si ninguna de las dos existe, la app usa memoria en RAM solo para desarrollo local.

Endpoints principales

- `POST /chat`
  - Descripción: Envía un mensaje al agente conversacional. El agente responde
    en lenguaje natural y puede generar actualizaciones de memoria.
  - Body (JSON): `{ "user_id": "string", "thread_id": "string", "message": "string" }`
  - Respuesta: `ChatResponse` (ver modelos en `app/schemas/api.py`).

- `GET /profile/{user_id}`
  - Descripción: Recupera el perfil completo acumulado del usuario.
  - Respuesta: `FullProfileResponse`.

- `DELETE /profile/{user_id}`
  - Descripción: Borra la memoria persistente del usuario (útil en pruebas).

- `PATCH /profile/{user_id}/profile-memory/{category}`
  - Descripción: Reemplaza exactamente un campo del perfil usando el texto
    provisto por el usuario (edición manual directa).
  - Body: `{ "text": "..." }`
  - Campos válidos (`category`): `interests`, `personality_traits` (o `traits`),
    `social_style`, `vibe_summary`, `favorite_environments`, `hobbies`, `dislikes`, `emotional_style`.

- `POST /profile/{user_id}/profile-memory/{category}/generate`  (nuevo)
  - Descripción: Genera y formatea el valor para un único campo a partir de un
    prompt corto del usuario. El endpoint usa la rutina `format_profile_field_with_model`
    en `app/agent/agent.py` y luego reemplaza el campo en la memoria persistente.
  - Body: `{ "text": "prompt o frase corta del usuario" }`
  - Comportamiento: El agente intentará interpretar y limpiar la entrada, devolverá
    un array para campos tipo lista (ej. `interests`) o una cadena para campos
    escalares (ej. `vibe_summary`). Si el modelo no puede producir un valor
    usable, retorna error 400.

Ejemplos rápidos (curl)

Enviar un mensaje al chat:

```bash
curl -X POST 'http://localhost:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","thread_id":"t1","message":"Me encantan las pelis de ciencia ficción"}'
```

Generar un `vibe_summary` desde un prompt:

```bash
curl -X POST 'http://localhost:8000/profile/alice/profile-memory/vibe_summary/generate' \
  -H 'Content-Type: application/json' \
  -d '{"text":"Creativa, tranquila y amante de las cafeterías acogedoras"}'
```

Reemplazar manualmente `interests` (edición directa):

```bash
curl -X PATCH 'http://localhost:8000/profile/alice/profile-memory/interests' \
  -H 'Content-Type: application/json' \
  -d '{"text":"Viajar, fotografía, cafés locales"}'
```

Notas de implementación
- El formateo y la generación de un solo campo se realiza en
  `app/agent/agent.py::format_profile_field_with_model`.
- La persistencia y operaciones atómicas sobre campos se manejan en
  `app/memory/memory_manager.py` (función `replace_profile_field`).
- El agente principal usa LangGraph y un adaptador opcional para Groq.

Desarrollo y pruebas
- Ejecute el servidor con `uvicorn` y pruebe los endpoints vía `curl` o en
  la UI swagger en `/docs`.
- Para integrar pruebas automáticas, puede añadir tests que llamen a la API
  (por ejemplo con `pytest` y `httpx`) y un fixture que prepare/limpie la
  memoria del usuario.

Contribuir
- Hacer un PR con cambios y agregar tests cuando corresponda.

Licencia
- (Opcional) Añada la licencia del proyecto aquí.

Si quieres, puedo añadir ejemplos de pruebas unitarias o integrar un script
de verificación local para ejecutar after `pip install -r requirements.txt`.
# Allora - AI Profile-Building Agent

Allora is a FastAPI + LangGraph service that builds a dating-app user profile through natural conversation.

The agent learns about the user's personality, interests, hobbies, social style, emotional style, favorite environments, dislikes, and conversation preferences. It stores that profile as long-term memory by `user_id`, while each `thread_id` keeps short-term conversation context.

## What The Agent Handles

The frontend handles fixed onboarding fields such as age, gender, dating goals, and profile settings.

Allora handles softer profile signals:

- Interests and hobbies
- Personality traits
- Social style
- Vibe summary
- Favorite environments
- Dislikes and turn-offs
- Emotional style
- Recent life context
- Conversation preferences

Every chat turn should extract useful signal and update memory when the user says something profile-relevant.

## Architecture

```text
FastAPI
  POST   /chat
  GET    /profile/{user_id}
  PATCH  /profile/{user_id}/profile-memory/{category}
  PATCH  /profile/{user_id}/profile-memory
  PATCH  /profile/{user_id}/preference-memory
  GET    /profile/{user_id}/match-payload
  DELETE /profile/{user_id}
  GET    /health

LangGraph
  load_memories -> profile_agent -> extract_and_save

Memory
  profile_memory
  context_memory
  preference_memory
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` and add your Groq key if you want the live model:

```text
GROQ_API_KEY=your_key_here
MONGODB_URI=your_mongodb_atlas_connection_string
MONGODB_DB_NAME=allora_agent
MONGODB_COLLECTION=memory
```

Start the API:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If port `8000` is busy, use another port:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Run the console chat:

```powershell
python docs/console_chat.py
```

If the API is on a non-default port:

```powershell
$env:ALLORA_BASE_URL="http://127.0.0.1:8001"
python docs/console_chat.py
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## API Reference

### POST `/chat`

Sends a natural conversation message to the profile-building agent.

Request:

```json
{
  "user_id": "user_abc123",
  "thread_id": "session-001",
  "message": "Me gusta bailar, cocinar y prefiero lugares tranquilos con poco ruido."
}
```

Response:

```json
{
  "assistant_message": "Bailar y cocinar dicen mucho de ti...",
  "memory_updates": {
    "profile_memory": {
      "interests": ["bailar", "cocinar"],
      "traits": [],
      "social_style": "prefers calm, intimate, low-noise environments",
      "vibe_summary": "Reflective person who prefers calm, intimate spaces.",
      "favorite_environments": ["lugares tranquilos", "poco ruido"],
      "hobbies": ["bailar", "cocinar"],
      "dislikes": [],
      "emotional_style": null
    },
    "context_memory": {
      "recent_topics": ["bailar", "cocinar", "lugares tranquilos", "poco ruido"],
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
    "profile_completion": 0.31,
    "should_continue": true,
    "turn_count": 1
  }
}
```

### GET `/profile/{user_id}`

Returns the full accumulated profile for one user.

Example:

```powershell
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/profile/user_abc123').json())"
```

Response shape:

```json
{
  "user_id": "user_abc123",
  "profile_memory": {
    "edad": null,
    "genero": null,
    "bio": null,
    "interests": [],
    "personality_traits": [],
    "social_style": null,
    "vibe_summary": null,
    "favorite_environments": [],
    "hobbies": [],
    "dislikes": [],
    "emotional_style": null,
    "location": null
  },
  "context_memory": {
    "recent_topics": [],
    "evolving_interests": [],
    "recent_life_changes": [],
    "recent_social_behavior": null,
    "current_mood_theme": null
  },
  "preference_memory": {
    "conversation_style": null,
    "prefers_short_questions": false,
    "depth_preference": null,
    "sensitive_topics": [],
    "response_length_preference": null
  },
  "match_preference_memory": {
    "edad_minima": null,
    "edad_maxima": null,
    "distancia_maxima_km": null,
    "genero_preferido": null
  },
  "profile_completion": 0.0
}
```

### PATCH `/profile/{user_id}/profile-memory`

Updates frontend-owned profile fields without calling the agent. Use this for
fixed onboarding data such as age, gender, bio and location, or for already
structured profile fields.

Request:

```json
{
  "edad": 28,
  "genero": "masculino",
  "bio": "Me gusta la naturaleza y planes tranquilos",
  "location": {
    "lat": 4.711,
    "lng": -74.072
  }
}
```

### PATCH `/profile/{user_id}/preference-memory`

Updates the dating/match filters collected by the frontend without the agent.
These values are stored separately from the agent's conversation preferences.

Request:

```json
{
  "edad_minima": 24,
  "edad_maxima": 34,
  "distancia_maxima_km": 30,
  "genero_preferido": "femenino"
}
```

### GET `/profile/{user_id}/match-payload`

Returns the exact payload expected by the match backend.

Response:

```json
{
  "user_id": "6a08df33e975899d2e2b238b",
  "profile_memory": {
    "edad": 28,
    "genero": "masculino",
    "bio": "Me gusta la naturaleza y planes tranquilos",
    "interests": ["senderismo", "musica", "cafe"],
    "personality_traits": ["tranquilo", "curioso"],
    "social_style": "Prefiere planes relajados",
    "vibe_summary": "Creativo y calmado",
    "favorite_environments": ["parques", "cafes"],
    "hobbies": ["fotografia", "senderismo"],
    "dislikes": ["ruido excesivo"],
    "emotional_style": "afectuoso y estable",
    "location": {
      "lat": 4.711,
      "lng": -74.072
    }
  },
  "preference_memory": {
    "edad_minima": 24,
    "edad_maxima": 34,
    "distancia_maxima_km": 30,
    "genero_preferido": "femenino"
  },
  "profile_completion": 1.0
}
```

## Guía Para El Frontend: Datos Sin Agente

El frontend debe recabar directamente los datos fijos de onboarding y filtros
de match. Estos datos no pasan por el agente conversacional.

### 1. Datos Del Perfil Del Usuario

Pide estos campos en el formulario de perfil/onboarding:

```json
{
  "edad": 28,
  "genero": "masculino",
  "bio": "Me gusta la naturaleza y planes tranquilos",
  "location": {
    "lat": 4.711,
    "lng": -74.072
  }
}
```

Endpoint:

```http
PATCH /profile/{user_id}/profile-memory
```

Ejemplo con `fetch`:

```js
await fetch(`${API_URL}/profile/${userId}/profile-memory`, {
  method: "PATCH",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    edad: 28,
    genero: "masculino",
    bio: "Me gusta la naturaleza y planes tranquilos",
    location: {
      lat: 4.711,
      lng: -74.072,
    },
  }),
});
```

Campos aceptados:

- `edad`: número entre 18 y 120.
- `genero`: string.
- `bio`: string.
- `location.lat`: número entre -90 y 90.
- `location.lng`: número entre -180 y 180.

También puedes mandar campos ya estructurados si el frontend los tiene:

```json
{
  "interests": ["senderismo", "musica", "cafe"],
  "hobbies": ["fotografia", "senderismo"],
  "personality_traits": ["tranquilo", "curioso"],
  "favorite_environments": ["parques", "cafes"],
  "social_style": "Prefiere planes relajados",
  "vibe_summary": "Creativo y calmado",
  "dislikes": ["ruido excesivo"],
  "emotional_style": "afectuoso y estable"
}
```

### 2. Preferencias De Match

Pide estos campos en la pantalla de preferencias/filtros:

```json
{
  "edad_minima": 24,
  "edad_maxima": 34,
  "distancia_maxima_km": 30,
  "genero_preferido": "femenino"
}
```

Endpoint:

```http
PATCH /profile/{user_id}/preference-memory
```

Ejemplo con `fetch`:

```js
await fetch(`${API_URL}/profile/${userId}/preference-memory`, {
  method: "PATCH",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    edad_minima: 24,
    edad_maxima: 34,
    distancia_maxima_km: 30,
    genero_preferido: "femenino",
  }),
});
```

Campos aceptados:

- `edad_minima`: número entre 18 y 120.
- `edad_maxima`: número entre 18 y 120.
- `distancia_maxima_km`: número mayor que 0.
- `genero_preferido`: string.

Regla importante: `edad_minima` no puede ser mayor que `edad_maxima`.

### 3. Payload Para El Backend De Match

Cuando el frontend ya haya enviado los datos directos y el agente haya
completado las partes conversacionales, pide el payload final:

```http
GET /profile/{user_id}/match-payload
```

Ejemplo con `fetch`:

```js
const response = await fetch(`${API_URL}/profile/${userId}/match-payload`);
const matchPayload = await response.json();
```

Ese `matchPayload` es el JSON que debes mandar al otro backend de match:

```js
await fetch(`${MATCH_API_URL}/match`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(matchPayload),
});
```

### Flujo Recomendado

1. El usuario llena onboarding fijo en el frontend.
2. El frontend llama `PATCH /profile/{user_id}/profile-memory`.
3. El usuario configura filtros de match.
4. El frontend llama `PATCH /profile/{user_id}/preference-memory`.
5. El usuario conversa con el agente para completar intereses, hobbies, vibe,
   dislikes, estilo social y estilo emocional.
6. El frontend llama `GET /profile/{user_id}/match-payload`.
7. El frontend envía ese payload al backend de match.

### PATCH `/profile/{user_id}/profile-memory/{category}`

Directly edits exactly one `profile_memory` category.

Use this endpoint when the user chooses to edit one profile section manually in the app. It does not run the chat agent and it does not update any other category. The selected category is replaced with a clean formatted value based only on the user's submitted text.

Valid categories:

- `interests`
- `personality_traits`
- `traits` - alias for `personality_traits`
- `social_style`
- `vibe_summary`
- `favorite_environments`
- `hobbies`
- `dislikes`
- `emotional_style`

Request:

```json
{
  "text": "lugares intimos y tranquilos, poco ruido, cafes"
}
```

Example for list categories:

```powershell
python -c "import httpx; r=httpx.patch('http://127.0.0.1:8000/profile/user_abc123/profile-memory/favorite_environments', json={'text':'lugares intimos y tranquilos, poco ruido, cafes'}); print(r.json())"
```

Response:

```json
{
  "user_id": "user_abc123",
  "category": "favorite_environments",
  "formatted_value": [
    "lugares intimos",
    "tranquilos",
    "poco ruido",
    "cafes"
  ],
  "profile_memory": {
    "interests": [],
    "personality_traits": [],
    "social_style": null,
    "vibe_summary": null,
    "favorite_environments": [
      "lugares intimos",
      "tranquilos",
      "poco ruido",
      "cafes"
    ],
    "hobbies": [],
    "dislikes": [],
    "emotional_style": null
  },
  "profile_completion": 0.04
}
```

Example for scalar categories:

```powershell
python -c "import httpx; r=httpx.patch('http://127.0.0.1:8000/profile/user_abc123/profile-memory/social_style', json={'text':'prefiero planes tranquilos, uno a uno, con poca presion social'}); print(r.json())"
```

Response:

```json
{
  "user_id": "user_abc123",
  "category": "social_style",
  "formatted_value": "Prefiero planes tranquilos, uno a uno, con poca presion social.",
  "profile_memory": {
    "interests": [],
    "personality_traits": [],
    "social_style": "Prefiero planes tranquilos, uno a uno, con poca presion social.",
    "vibe_summary": null,
    "favorite_environments": [],
    "hobbies": [],
    "dislikes": [],
    "emotional_style": null
  },
  "profile_completion": 0.12
}
```

### DELETE `/profile/{user_id}`

Deletes all memory for a user. Useful for testing.

Example:

```powershell
python -c "import httpx; print(httpx.delete('http://127.0.0.1:8000/profile/user_abc123').json())"
```

### GET `/health`

Health check.

```powershell
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health').json())"
```

## Memory Model

### Profile Memory

Long-term identity:

- `interests`
- `personality_traits`
- `social_style`
- `vibe_summary`
- `favorite_environments`
- `hobbies`
- `dislikes`
- `emotional_style`

### Context Memory

Recent life state:

- `recent_topics`
- `evolving_interests`
- `recent_life_changes`
- `recent_social_behavior`
- `current_mood_theme`

### Preference Memory

How the user likes to interact:

- `conversation_style`
- `prefers_short_questions`
- `depth_preference`
- `sensitive_topics`
- `response_length_preference`

### Match Preference Memory

Dating filters collected by the frontend and returned as `preference_memory`
by `/profile/{user_id}/match-payload`:

- `edad_minima`
- `edad_maxima`
- `distancia_maxima_km`
- `genero_preferido`

## Project Structure

```text
allora_agent/
  app/
    main.py
    agent/
      agent.py
    memory/
      memory_manager.py
      mongo_store.py
    schemas/
      api.py
      memory.py
  docs/
    console_chat.py
    example_conversations.py
  requirements.txt
  README.md
```

## Deployment

For Render or any container platform, use:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set environment variables in the platform dashboard:

```text
GROQ_API_KEY=your_key_here
```

Do not commit `.env` or `venv/`.
