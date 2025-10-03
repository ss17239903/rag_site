from sqlalchemy import create_engine, text
from typing import List, Dict, Any
import psycopg
import os
from collections import defaultdict

os.environ["PGCHANNELBIND"] = "disable"

class PostgresChatMemory:
    def __init__(self, db_url: str):
        self.conninfo = db_url

    def check_user(self, user_name: str):
        exist = True
        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id
                    FROM chat_user
                    WHERE user_name = %s
                    """
                    ,
                    (user_name,)
                )
                if cur.fetchone() is None:
                    exist = False

        return exist

    def create_user(self, user_name: str):
        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_user (user_name)
                    VALUES (%s)
                    ON CONFLICT (user_name) DO UPDATE SET user_name = EXCLUDED.user_name
                    RETURNING user_id
                    """
                    ,
                    (user_name,)
                )
                user_id = cur.fetchone()[0]
        return str(user_id)

    def save(self, thread_id: str, user_id: int, role: str, content: str, metadata: Dict[str, Any] = None):
        """save one chat message"""
        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (thread_id, user_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (thread_id, user_id, role, content, metadata)
                )
    def save_summary(self, user_id: int, thread_id: str, summary: str):
        """save one summary"""
        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_thread_summary (thread_id, user_id, summary)
                    VALUES (%s, %s, %s)
                    """,
                    (thread_id, user_id, summary)
                )
    def load_summary(self, user_id: int):
        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id, summary
                    FROM chat_thread_summary
                    WHERE user_id = %s
                    """,
                    (user_id,)
                )
                rows = cur.fetchall()
            summaries = defaultdict(list)
            for thread_id, summary in rows:
                summaries[thread_id] = summary
        return summaries 

    def load(self, thread_id: str) -> List[Dict[str, Any]]:
        """load all messages associated with a thread, ordered by time"""
        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, metadata, created_at
                    FROM chat_messages
                    WHERE thread_id = %s
                    ORDER BY created_at
                    """,
                    (thread_id,)
                )
                rows = cur.fetchall()

        msgs = []

        for r, c, m, a in rows:
            msgs.append({"role": r, "content": c})

        return msgs

    def load_user(self, user_id: int):
        """load all past messages from a user"""
        with psycopg.connect(self.conninfo) as conn:
           with conn.cursor() as cur:
               cur.execute(
                   """
                   SELECT thread_id, role, content, metadata, created_at
                   FROM chat_messages
                   WHERE user_id = %s
                   ORDER BY created_at
                   """,
                   (user_id,)
               )
               rows = cur.fetchall()

        thread_msgs = defaultdict(list)
        msgs = []
        for thread_id, role, content, m, a in rows:
            thread_msgs[thread_id].append({"content": content})
            msgs.append({"role": role, "content": content})

        return thread_msgs, msgs


    def delete_session(self, session_id: str):
        """Delete a chat session and all messages"""
        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
    #
    # def load_formatted(self, session_id: str) -> str:
    #     """Return all messages as a single string for LLM context"""
    #     history = self.load(session_id)
    #     formatted = []
    #     for msg in history:
    #         formatted.append(f"{msg['role'].capitalize()}: {msg['content']}")
    #     return "\n".join(formatted)
