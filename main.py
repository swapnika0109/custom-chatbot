import streamlit as st
import requests
import logging

def rasa_response(prompt):
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = requests.post(
            "http://localhost:5005/webhooks/rest/webhook", 
            json={"message": prompt}
        )
        rasa_response = response.json()
        logging.info('received response ', rasa_response)
        # Add assistant response to chat history
        if rasa_response:
            response_text = rasa_response[0].get("text", "I'm not sure how to respond to that.")
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            logging.info('received response ', st.session_state.messages)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I'm not sure how to respond to that."})
            logging.info('received response ', st.session_state.messages)

    except requests.exceptions.RequestException as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error connecting to Rasa: {e}"})

def main():
    logging.basicConfig(level=logging.INFO, format=' %(levelname)s - %(message)s')
    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Fashion Assistant Chatbot")
    
    # Display chat history
    for message in st.session_state.messages:
        role = "👤 You:" if message["role"] == "user" else "🤖 Assistant:"
        st.text(f"{role} {message['content']}")

    # Chat input
    prompt = st.text_input("What's your fashion question?", key="user_input")
    logging.info('received prompt ', prompt)
    if prompt:
        st.button("Send", on_click=lambda: rasa_response(prompt))
        
        

if __name__ == "__main__":
    main()
