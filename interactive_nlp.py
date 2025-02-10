from rasa_bot.nlp.processor import NLPProcessor

def interactive_test():
    print("Initializing NLP Processor...")
    nlp = NLPProcessor()
    print("\nNLP Processor ready! Enter your fashion-related questions (type 'quit' to exit)")
    
    while True:
        print("\n" + "="*50)
        text = input("\nEnter your question: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not text:
            continue
            
        print("\nSelect analysis type:")
        print("1. Complete analysis")
        print("2. Preprocess text")
        print("3. Extract entities")
        print("4. Classify intent")
        print("5. Get fashion attributes")
        print("6. Generate response")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        try:
            if choice == '1':
                result = nlp.process_fashion_query(text)
                for key, value in result.items():
                    print(f"\n{key.capitalize()}:")
                    print(value)
                    
            elif choice == '2':
                print("\nPreprocessed text:")
                print(nlp.preprocess_text(text))
                
            elif choice == '3':
                print("\nExtracted entities:")
                print(nlp.extract_entities(text))
                
            elif choice == '4':
                print("\nIntent classification:")
                print(nlp.classify_intent(text))
                
            elif choice == '5':
                print("\nFashion attributes:")
                print(nlp.classify_fashion_attributes(text))
                
            elif choice == '6':
                print("\nGenerated response:")
                print(nlp.generate_response(text))
                
            else:
                print("\nInvalid choice! Please select 1-6")
                
        except Exception as e:
            print(f"\nError processing request: {str(e)}")

if __name__ == "__main__":
    interactive_test() 