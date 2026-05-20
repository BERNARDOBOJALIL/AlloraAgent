"""
Mongo-backed key/value store for Allora long-term memory.

The memory manager only needs a tiny subset of LangGraph's store API:
search(namespace), put(namespace, key, value), and delete(namespace, key).
This adapter keeps that interface while persisting documents in MongoDB Atlas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple


Namespace = Tuple[str, str]


@dataclass(frozen=True)
class MongoStoreItem:
    key: str
    value: Dict[str, Any]


class MongoMemoryStore:
    def __init__(
        self,
        uri: str,
        database_name: str = "allora_agent",
        collection_name: str = "memory",
        server_selection_timeout_ms: int = 5000,
    ) -> None:
        try:
            from pymongo import ASCENDING, MongoClient
            from pymongo.collection import Collection
        except ImportError as exc:  # pragma: no cover - depends on installed deps
            raise RuntimeError(
                "MongoDB persistence requires `pymongo`. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._client = MongoClient(uri, serverSelectionTimeoutMS=server_selection_timeout_ms)
        self._client.admin.command("ping")
        self._collection: Collection = self._client[database_name][collection_name]
        self._collection.create_index(
            [("namespace_kind", ASCENDING), ("user_id", ASCENDING), ("key", ASCENDING)],
            unique=True,
        )

    @staticmethod
    def _namespace_parts(namespace: Iterable[str]) -> Namespace:
        parts = tuple(namespace)
        if len(parts) != 2:
            raise ValueError("Expected namespace shape: (kind, user_id).")
        return parts[0], parts[1]

    @staticmethod
    def _document_id(kind: str, user_id: str, key: str) -> str:
        return f"{kind}:{user_id}:{key}"

    def search(self, namespace: Namespace) -> List[MongoStoreItem]:
        kind, user_id = self._namespace_parts(namespace)
        cursor = self._collection.find(
            {"namespace_kind": kind, "user_id": user_id},
            {"_id": 0, "key": 1, "value": 1},
        ).sort("key", 1)
        return [
            MongoStoreItem(key=doc["key"], value=doc.get("value", {}))
            for doc in cursor
        ]

    def put(self, namespace: Namespace, key: str, value: Dict[str, Any]) -> None:
        kind, user_id = self._namespace_parts(namespace)
        now = datetime.now(timezone.utc)
        self._collection.update_one(
            {"_id": self._document_id(kind, user_id, key)},
            {
                "$set": {
                    "namespace_kind": kind,
                    "user_id": user_id,
                    "key": key,
                    "value": value,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def delete(self, namespace: Namespace, key: str) -> None:
        kind, user_id = self._namespace_parts(namespace)
        self._collection.delete_one({"_id": self._document_id(kind, user_id, key)})
