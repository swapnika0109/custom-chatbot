import logging
from functools import wraps
from typing import Callable, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotError(Exception):
    """Base class for chatbot exceptions"""
    pass

class ErrorHandler:
    @staticmethod
    def handle_errors(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return {
                    'error': True,
                    'message': "I apologize, but I encountered an error. Please try again.",
                    'error_type': type(e).__name__
                }
        return wrapper 