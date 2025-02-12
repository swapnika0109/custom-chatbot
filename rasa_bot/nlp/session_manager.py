from datetime import datetime, timedelta

class SessionManager:
    def __init__(self, session_timeout: int = 30):
        self.sessions = {}
        self.timeout = timedelta(minutes=session_timeout)

    def start_session(self, user_id: str):
        self.sessions[user_id] = {
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'interaction_count': 0
        }

    def update_session(self, user_id: str):
        if user_id in self.sessions:
            self.sessions[user_id]['last_activity'] = datetime.now()
            self.sessions[user_id]['interaction_count'] += 1

    def is_session_active(self, user_id: str) -> bool:
        if user_id not in self.sessions:
            return False
        return datetime.now() - self.sessions[user_id]['last_activity'] < self.timeout 