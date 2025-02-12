from typing import Dict, Any
from .nlp.enhanced_processor import EnhancedNLPProcessor
from .nlp.conversation_manager import ConversationManager
from .nlp.session_manager import SessionManager
from .nlp.intent_classifier import IntentClassifier
from .utils.error_handler import ErrorHandler, ChatbotError
from .utils.rate_limiter import RateLimiter
from .utils.config_manager import ConfigManager

class FashionChatbot:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.nlp_processor = EnhancedNLPProcessor(
            db_path=self.config.get('database.path')
        )
        self.conversation_manager = ConversationManager(
            max_history=self.config.get('session.max_history', 5)
        )
        self.session_manager = SessionManager(
            session_timeout=self.config.get('session.timeout_minutes', 30)
        )
        self.intent_classifier = IntentClassifier()
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.get('rate_limits.requests_per_minute', 30)
        )

    @ErrorHandler.handle_errors
    def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        # Check rate limit
        if not self.rate_limiter.can_process(user_id):
            raise ChatbotError("Rate limit exceeded. Please try again later.")

        # Start or update session
        if not self.session_manager.is_session_active(user_id):
            self.session_manager.start_session(user_id)
        else:
            self.session_manager.update_session(user_id)

        # Classify intent
        intent, confidence = self.intent_classifier.classify_intent(message)

        # Get conversation history
        history = self.conversation_manager.get_history(user_id)

        # Generate response
        response, response_confidence = self.nlp_processor.generate_enhanced_response(
            message,
            context=history
        )

        # Store interaction
        self.conversation_manager.add_interaction(user_id, message, response)

        return {
            'response': response,
            'intent': intent,
            'intent_confidence': confidence,
            'response_confidence': response_confidence,
            'session_active': True
        }

    def cleanup(self):
        self.nlp_processor.cleanup()

# Example usage
if __name__ == "__main__":
    chatbot = FashionChatbot()
    
    # Simulate a conversation
    user_id = "user123"
    messages = [
        "Hi, I need help with fashion advice",
        "Can I wear a wool sweater in summer?",
        "What about a cotton t-shirt instead?",
        "Thanks, goodbye!"
    ]
    
    for message in messages:
        result = chatbot.process_message(user_id, message)
        print(f"\nUser: {message}")
        print(f"Bot: {result['response']}")
        print(f"Intent: {result['intent']} ({result['intent_confidence']:.2f})")
    
    chatbot.cleanup() 