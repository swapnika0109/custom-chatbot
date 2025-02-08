import streamlit as st
import requests

def main():
    st.title("Fashion Assistant Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role = "👤 You:" if message["role"] == "user" else "🤖 Assistant:"
        st.text(f"{role} {message['content']}")

    # Chat input
    prompt = st.text_input("What's your fashion question?", key="user_input")
    
    if st.button("Send") and prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response from Rasa
        rasa_response = requests.post(
            "http://localhost:5005/webhooks/rest/webhook", 
            json={"message": prompt}
        ).json()
        
        # Add assistant response to chat history
        if rasa_response:
            response_text = rasa_response[0].get("text", "I'm not sure how to respond to that.")
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I'm not sure how to respond to that."})
        
        # Rerun to update the chat display
        st.experimental_rerun()

if __name__ == "__main__":
    main()
