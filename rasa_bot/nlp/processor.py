import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class NLPProcessor:
    def __init__(self):
        # Load models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Set pad token to eos token since GPT2 doesn't have a pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Force CPU usage to avoid CUDA memory issues
        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map=None,  # Don't use auto device mapping
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to('cpu')  # Explicitly move to CPU
        
        self.nlp = spacy.load("en_core_web_sm")
        
        # Force SentenceTransformer to use CPU
        self.sentence_transformer = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu'  # Explicitly set device to CPU
        )
        
        # Initialize FAISS with correct dimension
        self.vector_dim = 384  # MiniLM-L6-v2 produces 384-dimensional embeddings
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.stored_queries = []

    def get_text_embedding(self, text):
        """Convert text into a fixed-size embedding (random example here)."""
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding.astype('float32')
        except Exception as e:
            print(f"Error generating embedding: {e}")
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(self.vector_dim).astype('float32')
    

    def preprocess_text(self, text):
        return text.lower().strip()

    def extract_fashion_entities(self, text):
        doc = self.nlp(text)
        entities = {"colors": [], "fabrics": [], "seasons": []}
        
        color_list = ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray"]
        fabric_list = ["cotton", "silk", "denim", "wool", "linen", "leather"]
        season_list = ["summer", "winter", "spring", "autumn", "fall"]
        
        for token in doc:
            if token.text.lower() in color_list:
                entities["colors"].append(token.text.lower())
            elif token.text.lower() in fabric_list:
                entities["fabrics"].append(token.text.lower())
            elif token.text.lower() in season_list:
                entities["seasons"].append(token.text.lower())
        
        return entities

    def retrieve_from_faiss(self, query_embedding):
        if self.index.ntotal == 0:
            return None

        _, I = self.index.search(np.array([query_embedding]), 1)
        best_match_index = I[0][0]

        if best_match_index >= 0:
            return self.stored_queries[best_match_index]
        return None

    def generate_response(self, prompt):
        try:
            eng_prompt = f"""Fashion advice for specific clothing items:
            Q: Can I wear a thick sweater in summer?
            A: No, a thick sweater is too warm for summer. Choose lightweight, breathable fabrics instead.

            Q: {prompt}
            A: Let me give specific advice about this clothing choice."""

            # Tokenize with smaller batch size
            inputs = self.tokenizer(
                eng_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
                add_special_tokens=True
            )

            # Move inputs to the same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate with adjusted parameters
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=32,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,  # Increased slightly for more variety
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    max_new_tokens=20,
                    repetition_penalty=1.8  # Increased to avoid copying example
                )

            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the response part after the last "A:"
            response_parts = decoded_output.split("A:")
            if len(response_parts) > 1:
                response = response_parts[-1].strip()
                response = response.split('\n')[0].split('|')[0].strip()
                if "Let me give specific advice" in response:  # Remove the prompt text if it appears
                    response = response.replace("Let me give specific advice about this clothing choice.", "").strip()
                return response

            return "I apologize, I couldn't generate appropriate fashion advice."

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return "I apologize, but I couldn't generate a response."

    def fashion_chatbot(self, user_query):
        try:
            # Step 1: Preprocess input
            clean_query = self.preprocess_text(user_query)
            print(f"Preprocessed query: {clean_query}")

            # Step 2: Extract fashion-related entities
            extracted_info = self.extract_fashion_entities(clean_query)
            print(f"Extracted Entities: {extracted_info}")

            # Step 3: Generate text embedding
            query_embedding = self.get_text_embedding(clean_query)
            print(f"Generated embedding shape: {query_embedding.shape}")

            # Step 4: Try retrieving from FAISS
            retrieved_response = self.retrieve_from_faiss(query_embedding)
            if retrieved_response:
                print("Retrieved from Vector Store")
                return retrieved_response

            # Step 5: Generate response
            print("Generating new response...")
            response = self.generate_response(clean_query)
            
            # Step 6: Store in FAISS
            self.index.add(np.array([query_embedding]))
            self.stored_queries.append(response)

            return response  # Return response without printing

        except Exception as e:
            print(f"Error in fashion_chatbot: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            return "I apologize, but an error occurred while processing your request."

# Example usage (optional)
if __name__ == "__main__":
    processor = NLPProcessor()
    
    test_queries = [
        "I am wearing a wool dress in summer, is it okay?",
        "Can I wear a leather jacket in winter?",
        "Are white pants good for summer?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("Bot:", processor.fashion_chatbot(query))
