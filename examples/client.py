import requests
import json

def chat_with_bot(message: str, user_id: str = "test_user"):
    url = "http://localhost:8000/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "user_id": user_id,
        "message": message
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

if __name__ == "__main__":
    while True:
        message = input("You: ")
        if message.lower() in ['quit', 'exit']:
            break
            
        response = chat_with_bot(message)
        print(f"Bot: {response['response']}")
        print(f"Intent: {response['intent']} (confidence: {response['confidence']:.2f})") 