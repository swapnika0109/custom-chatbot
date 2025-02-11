from typing import Text, Dict, Any, List
import spacy
import re
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from spacy.matcher import Matcher
import matplotlib.colors as mcolors

from sentence_transformers import SentenceTransformer
import numpy as np

class NLPProcessor:
    def __init__(self):
        # Initialize models
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Set up padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # GPT-2 requires left padding
        
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Add zero-shot classifier
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Add text classifier
        self.text_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def preprocess_text(self, text: Text) -> Text:
        """
        1. Preprocessing Pipeline
        - Lowercase conversion
        - Remove special characters
        - Remove extra whitespace
        - Tokenization
        - Remove stopwords
        - Lemmatization
        """
        # Lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # SpaCy processing
        doc = self.nlp(text)
        
        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)

    def extract_entities(self, text: Text) -> Dict[str, List[Text]]:
        """
        2. Entity Recognition
        - Fashion-specific entities
        - Named Entity Recognition (NER)
        - Custom entity patterns
        """
        doc = self.nlp(self.preprocess_text(text))
        
        # Initialize entities dictionary
        entities = {
            'clothing': [],
            'occasions': [],
            'seasons': [],
            'colors': [],
            'materials': [],
            'brands': []
        }
        
        # Define patterns
        clothing_patterns = [[{'LOWER': word}] for word in ['dress', 'shirt', 'pants', 'skirt']]
        occasions_patterns = [[{'LOWER': word}] for word in ['party', 'formal', 'casual', 'wedding']]
        seasons_patterns = [[{'LOWER': word}] for word in ['summer', 'winter', 'spring', 'fall']]
        colors_patterns = [[{'LOWER': color.lower()}] for color in mcolors.CSS4_COLORS.keys()]
        materials_patterns = [[{'LOWER': word}] for word in ['cotton', 'silk', 'wool', 'leather']]

        # Create matcher and add patterns
        matcher = Matcher(self.nlp.vocab)
        
        # Add patterns with proper labels
        matcher.add("CLOTHING", clothing_patterns)
        matcher.add("OCCASIONS", occasions_patterns)
        matcher.add("SEASONS", seasons_patterns)
        matcher.add("COLORS", colors_patterns)
        matcher.add("MATERIALS", materials_patterns)
        
        # Find matches
        matches = matcher(doc)
        
        # Process matches
        for match_id, start, end in matches:
            matched_span = doc[start:end].text
            # Convert the match_id to string category name
            category = self.nlp.vocab.strings[match_id].upper()
            if category in entities:
                entities[category].append(matched_span)
        
        # Add SpaCy NER entities
        for ent in doc.ents:
            if ent.label_ == 'ORG':  # Potential brand names
                entities['brands'].append(ent.text)
                
        return entities

    def classify_intent(self, text: Text) -> Dict[str, float]:
        """
        3. Classification Pipeline using zero-shot and text classification
        """
        # Define possible fashion intents
        candidate_intents = [
            "outfit recommendation",
            "style advice",
            "fashion trend inquiry",
            "product search",
            "color coordination",
            "size guidance",
            "brand recommendations"
        ]
        
        # Zero-shot classification for flexible intent recognition
        zero_shot_result = self.zero_shot_classifier(
            text,
            candidate_intents,
            multi_label=True
        )
        
        # Text classification for sentiment and urgency
        sentiment_result = self.text_classifier(text)

       
        
        return {
            'intents': {
                label: score 
                for label, score in zip(
                    zero_shot_result['labels'], 
                    zero_shot_result['scores']
                )
            },
            'sentiment': sentiment_result[0]
        }

    def classify_fashion_attributes(self, text: Text) -> Dict[str, List[Dict[str, float]]]:
        """
        Zero-shot classification for fashion attributes
        """
        attribute_categories = {
            'style': [
                "casual", "formal", "business", "party", "sporty"
            ],
            'fit': [
                "loose", "tight", "regular", "oversized", "slim"
            ],
            'occasion': [
                "work", "wedding", "date", "everyday", "workout"
            ]
        }
        
        results = {}
        for category, labels in attribute_categories.items():
            classification = self.zero_shot_classifier(
                text,
                labels,
                multi_label=True
            )
            results[category] = [
                {'label': label, 'score': score}
                for label, score in zip(
                    classification['labels'],
                    classification['scores']
                )
            ]
        
        return results

    def get_embeddings(self, text: Text) -> np.ndarray:
        """
        4. Feature Extraction
        - Text to vector conversion
        - Semantic embeddings
        """
        # Preprocess text first
        processed_text = self.preprocess_text(text)
        
        # Get embeddings using Sentence Transformers
        embeddings = self.embedding_model.encode([processed_text])[0]
        return embeddings

   

    def fine_tune_model(self, training_data: List[Dict[str, Text]]):
        """
        5. LLM Fine-tuning
        - Custom fashion domain adaptation
        - Training on fashion-specific data
        """
        # TODO: Implement fine-tuning logic
        # This would typically involve:
        # 1. Preparing fashion-specific training data
        # 2. Fine-tuning the base model
        # 3. Saving the fine-tuned model
        pass
        

    def generate_response(self, prompt: Text) -> Text:
        """
        6. Text Generation
        - Fashion-specific response generation
        - Context-aware responses
        """
        # Create a more focused fashion prompt
        fashion_prompt = f"As a fashion expert, provide a helpful response to: {prompt}\n\nResponse:"
        
        # Generate response using the model
        inputs = self.tokenizer(fashion_prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            **inputs,
            max_length=150,  # Increased for more detailed responses
            min_length=50,   # Ensure responses aren't too short
            num_return_sequences=1,
            temperature=0.8,  # Slightly increased for more creative responses
            top_p=0.9,       # Nucleus sampling
            do_sample=True,  # Enable sampling
            no_repeat_ngram_size=2,  # Avoid repetition
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and clean up the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt if it appears in the response
        response = response.replace(fashion_prompt, "").strip()
        
        return response

    def process_fashion_query(self, text: Text) -> Dict[str, Any]:
        """
        Complete pipeline execution
        """
        return {
            'preprocessed': self.preprocess_text(text),
            'entities': self.extract_entities(text),
            'intent': self.classify_intent(text),
            'embeddings': self.get_embeddings(text),
            'response': self.generate_response(text)
        }