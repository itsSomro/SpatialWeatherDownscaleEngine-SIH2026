from typing import Dict, List, Any, Optional

class ConversationMemory:
    """In-memory thread-safe conversation history storage."""

    def __init__(self, max_history_per_thread: int = 20):
        self.threads: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history_per_thread

    def get_messages(self, thread_id: Optional[str] = None) -> List[Dict[str, str]]:
        t_id = thread_id or "default"
        return list(self.threads.get(t_id, []))

    def add_message(self, role: str, content: str, thread_id: Optional[str] = None) -> None:
        t_id = thread_id or "default"
        if t_id not in self.threads:
            self.threads[t_id] = []
        self.threads[t_id].append({"role": role, "content": content})
        if len(self.threads[t_id]) > self.max_history:
            self.threads[t_id] = self.threads[t_id][-self.max_history:]

    def clear(self, thread_id: Optional[str] = None) -> None:
        t_id = thread_id or "default"
        if t_id in self.threads:
            self.threads[t_id] = []


memory = ConversationMemory()

