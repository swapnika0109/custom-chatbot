import time
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, start_http_server

class MonitoringSystem:
    def __init__(self, metrics_port: int = 9090):
        self.request_counter = Counter('chatbot_requests_total', 'Total chat requests')
        self.response_time = Histogram('chatbot_response_time_seconds', 'Response time in seconds')
        self.error_counter = Counter('chatbot_errors_total', 'Total errors')
        
        # Start Prometheus metrics server
        start_http_server(metrics_port)
        
        # Setup logging
        logging.basicConfig(
            filename='chatbot.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_request(self, user_id: str, message: str, response: str, duration: float):
        self.request_counter.inc()
        self.response_time.observe(duration)
        self.logger.info(f"Request from {user_id}: {message[:50]}...")

    def log_error(self, error_type: str, error_message: str):
        self.error_counter.inc()
        self.logger.error(f"Error {error_type}: {error_message}") 