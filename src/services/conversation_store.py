"""
Persistent conversation storage using SQLite.

Stores conversation history for multi-turn image generation,
allowing conversations to persist across server restarts.
"""

import json
import logging
import os
import sqlite3
import stat
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".imagen-mcp" / "conversations.db"


class ConversationStore:
    """
    SQLite-based persistent conversation storage.

    Stores conversation history with provider-specific metadata,
    enabling multi-turn image generation workflows.

    Uses a persistent connection to avoid the overhead of opening and
    closing an SQLite file on every operation (~0.5-2 ms each).
    """

    def __init__(self, db_path: Path | str | None = None):
        """
        Initialize the conversation store.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.imagen-mcp/conversations.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

        # Restrict database file to owner-only read/write (0o600) so that
        # other users on the system cannot read conversation history.
        if self.db_path.exists():
            os.chmod(self.db_path, stat.S_IRUSR | stat.S_IWUSR)

    def _get_persistent_connection(self) -> sqlite3.Connection:
        """Return the persistent connection, creating it on first use."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            # WAL mode gives better concurrent read performance
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get the persistent database connection."""
        yield self._get_persistent_connection()

    def close(self) -> None:
        """Close the persistent connection (call on shutdown)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_base64 TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id);

                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
            """)
            conn.commit()

    def create_conversation(self, conversation_id: str, provider: str) -> None:
        """
        Create a new conversation.

        Args:
            conversation_id: Unique conversation ID
            provider: Provider name (openai, gemini)
        """
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations (id, provider, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, provider, now, now),
            )
            conn.commit()

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: Any,
        image_base64: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation to add to
            role: Message role (user, assistant, system)
            content: Message content (will be JSON-serialized if not string)
            image_base64: Optional base64-encoded image
            metadata: Optional metadata dict
        """
        now = datetime.now().isoformat()

        # Serialize content if not string
        if not isinstance(content, str):
            content = json.dumps(content)

        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            # Add message
            conn.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content, image_base64, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, image_base64, metadata_json, now),
            )

            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: Conversation to retrieve
            limit: Max messages to return (newest first if limited)

        Returns:
            List of message dicts with role, content, image_base64, metadata
        """
        with self._get_connection() as conn:
            query = """
                SELECT role, content, image_base64, metadata, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
            """
            if limit:
                # Get total count to slice from end
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                total = count_row[0] if count_row else 0
                if total > limit:
                    query = """
                        SELECT role, content, image_base64, metadata, created_at
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY id ASC
                        LIMIT ? OFFSET ?
                    """
                    rows = conn.execute(query, (conversation_id, limit, total - limit)).fetchall()
                else:
                    rows = conn.execute(query, (conversation_id,)).fetchall()
            else:
                rows = conn.execute(query, (conversation_id,)).fetchall()

            messages = []
            for row in rows:
                msg: dict[str, Any] = {
                    "role": row["role"],
                    "content": self._parse_json_or_string(row["content"]),
                    "created_at": row["created_at"],
                }
                if row["image_base64"]:
                    msg["image_base64"] = row["image_base64"]
                if row["metadata"]:
                    msg["metadata"] = json.loads(row["metadata"])
                messages.append(msg)

            return messages

    def _parse_json_or_string(self, value: str) -> Any:
        """Parse JSON if possible, otherwise return string."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """
        Get conversation metadata.

        Args:
            conversation_id: Conversation to retrieve

        Returns:
            Dict with id, provider, created_at, updated_at, or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()

            if row:
                return {
                    "id": row["id"],
                    "provider": row["provider"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            return None

    def list_conversations(
        self,
        provider: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        List recent conversations.

        Uses a single query with a window function to fetch message
        counts and last-message previews — avoids the previous N+1
        pattern that issued one extra SELECT per conversation.

        Args:
            provider: Optional provider filter
            limit: Max conversations to return

        Returns:
            List of conversation summaries
        """
        with self._get_connection() as conn:
            # Single query: join + window function replaces N+1
            base_query = """
                SELECT
                    c.id,
                    c.provider,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count,
                    (
                        SELECT content FROM messages m2
                        WHERE m2.conversation_id = c.id
                        ORDER BY m2.id DESC LIMIT 1
                    ) AS last_content
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
            """
            if provider:
                base_query += " WHERE c.provider = ?"
                params: tuple[Any, ...] = (provider, limit)
            else:
                params = (limit,)

            base_query += """
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
            """
            rows = conn.execute(base_query, params).fetchall()

            conversations = []
            for row in rows:
                last_content_str = "No messages"
                raw = row["last_content"]
                if raw:
                    content = self._parse_json_or_string(raw)
                    if isinstance(content, str):
                        last_content_str = content[:50]
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                last_content_str = part.get("text", "")[:50]
                                break
                    elif isinstance(content, dict) and "prompt" in content:
                        last_content_str = content["prompt"][:50]

                conversations.append(
                    {
                        "id": row["id"],
                        "provider": row["provider"],
                        "message_count": row["message_count"],
                        "last_message": last_content_str,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                )

            return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and its messages.

        Args:
            conversation_id: Conversation to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            # Check existence
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()

            if not exists:
                return False

            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return True

    def get_last_image(self, conversation_id: str) -> str | None:
        """
        Get the last image from a conversation.

        Args:
            conversation_id: Conversation to search

        Returns:
            Base64-encoded image or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT image_base64 FROM messages
                WHERE conversation_id = ? AND image_base64 IS NOT NULL
                ORDER BY id DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()

            return row["image_base64"] if row else None

    def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Delete conversations older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of conversations deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            # Get IDs to delete
            rows = conn.execute(
                "SELECT id FROM conversations WHERE updated_at < ?", (cutoff,)
            ).fetchall()

            if not rows:
                return 0

            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids))

            conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", ids)
            conn.commit()

            return len(ids)


# Singleton instance
_store: ConversationStore | None = None


def get_conversation_store(db_path: Path | str | None = None) -> ConversationStore:
    """Get the singleton conversation store instance."""
    global _store
    if _store is None:
        _store = ConversationStore(db_path)
    return _store
