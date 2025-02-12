import pytest
from rasa_bot.chatbot import FashionChatbot
from rasa_bot.utils.error_handler import ChatbotError

@pytest.fixture
def chatbot():
    return FashionChatbot()

def test_basic_response(chatbot):
    response = chatbot.process_message("test_user", "Can I wear a wool sweater in summer?")
    assert response is not None
    assert 'response' in response
    assert 'intent' in response
    assert 'response_confidence' in response

def test_rate_limiting(chatbot):
    # Test rapid requests
    for _ in range(31):  # Assuming 30 requests per minute limit
        response = chatbot.process_message("test_user", "test message")
    
    with pytest.raises(ChatbotError) as exc_info:
        chatbot.process_message("test_user", "test message")
    assert "Rate limit exceeded" in str(exc_info.value)

def test_intent_classification(chatbot):
    response = chatbot.process_message("test_user", "Hello!")
    assert response['intent'] == 'greeting' 