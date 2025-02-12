from typing import List, Dict
from datetime import datetime

class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.conversations: Dict[str, List[Dict]] = {}
        self.max_history = max_history

    def add_interaction(self, user_id: str, message: str, response: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            
        self.conversations[user_id].append({
            'timestamp': datetime.now(),
            'user_message': message,
            'bot_response': response
        })
        
        # Keep only recent history
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id].pop(0)

    def get_history(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])

    def clear_history(self, user_id: str):
        self.conversations[user_id] = [] 