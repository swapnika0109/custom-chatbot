import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import sqlite3
import json
from datetime import datetime
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNLPProcessor:
    def __init__(self, db_path: str = 'fashion_kb.db'):
        # Initialize models (similar to original)
        self._init_models()
        # Initialize database
        self._init_database(db_path)
        # Initialize cache
        self.query_cache = {}
        self.cache_size = 1000
        
    def _init_models(self):
        """Initialize all required models"""
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # TODO
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map=None,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to('cpu')
        
        self.nlp = spacy.load("en_core_web_sm")
        
        self.sentence_transformer = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu'
        )
        
        # Initialize FAISS with IVF for better scaling
        self.vector_dim = 384
        self.quantizer = faiss.IndexFlatL2(self.vector_dim)
        nlist = 100  # number of clusters
        self.index = faiss.IndexIVFFlat(self.quantizer, self.vector_dim, nlist)
        self.index.train(np.random.rand(1000, self.vector_dim).astype('float32'))

    def _init_database(self, db_path: str):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                query TEXT,
                response TEXT,
                embedding BLOB,
                timestamp DATETIME,
                feedback_score INTEGER DEFAULT 0,
                version TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_index (
                conversation_id INTEGER,
                entity_type TEXT,
                entity_value TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        self.conn.commit()

    def hybrid_search(self, query: str, k: int = 3) -> List[Dict]:
        """Combine semantic and keyword search"""
        # Check cache first
        cache_key = f"{query}_{k}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Get semantic search results
        query_embedding = self.get_text_embedding(query)
        semantic_results = self._semantic_search(query_embedding, k)
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, k)
        
        # Combine and rank results
        combined_results = self._merge_search_results(semantic_results, keyword_results)
        
        # Update cache
        self.query_cache[cache_key] = combined_results
        if len(self.query_cache) > self.cache_size:
            self.query_cache.pop(next(iter(self.query_cache)))
        
        return combined_results

    def _semantic_search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Perform semantic search using FAISS"""
        self.index.nprobe = 10  # Number of clusters to visit during search
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                conversation = self._get_conversation_by_id(idx)
                if conversation:
                    results.append({
                        'id': idx,
                        'score': float(1 / (1 + dist)),
                        'data': conversation
                    })
        return results

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """Perform keyword-based search"""
        # Extract entities from query
        entities = self.extract_fashion_entities(query)
        
        # Build SQL query
        sql = '''
            SELECT DISTINCT c.* 
            FROM conversations c
            JOIN entity_index e ON c.id = e.conversation_id
            WHERE 1=1
        '''
        params = []
        for entity_type, values in entities.items():
            if values:
                placeholders = ','.join('?' * len(values))
                sql += f' AND EXISTS (SELECT 1 FROM entity_index e2 WHERE e2.conversation_id = c.id AND e2.entity_type = ? AND e2.entity_value IN ({placeholders}))'
                params.extend([entity_type] + values)
        
        sql += ' LIMIT ?'
        params.append(k)
        
        return [{'id': row[0], 'data': row} for row in self.cursor.execute(sql, params).fetchall()]

    def _merge_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Merge and rank results from both search methods"""
        merged = defaultdict(float)
        
        # Combine scores
        for result in semantic_results:
            merged[result['id']] += result['score'] * 0.7  # Weight for semantic search
            
        for result in keyword_results:
            merged[result['id']] += 0.3  # Weight for keyword match
            
        # Get final sorted results
        return sorted(
            [{'id': k, 'score': v, 'data': self._get_conversation_by_id(k)} 
             for k, v in merged.items()],
            key=lambda x: x['score'],
            reverse=True
        )

    def generate_enhanced_response(self, query: str) -> Tuple[str, float, bool]:
        """Generate response with context and confidence score"""
        try:
            # 1. Search for similar past conversations
            contexts = self.hybrid_search(query)
            
            # 2. Check context relevance
            has_relevant_context = self._check_context_relevance(query, contexts)
            
            if has_relevant_context:
                # Use existing flow with context
                context_text = self._prepare_context(contexts)
                response = self._generate_with_context(query, context_text)
            else:
                # Generate response without context
                response = self._generate_without_context(query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(response, contexts, has_relevant_context)
            
            # Store good responses for future use
            if confidence > 0.7:
                self._store_conversation(query, response)
            
            return response, confidence, has_relevant_context
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I couldn't generate a response.", 0.0, False

    def _check_context_relevance(self, query: str, contexts: List[Dict]) -> bool:
        """Check if retrieved contexts are relevant to the query"""
        if not contexts:
            return False
        
        query_embedding = self.get_text_embedding(query)
        relevance_threshold = 0.6  # Adjustable threshold
        
        relevant_contexts = 0
        for context in contexts[:3]:  # Check top 3 contexts
            # Calculate semantic similarity
            context_text = f"{context['data']['query']} {context['data']['response']}"
            context_embedding = self.get_text_embedding(context_text)
            
            similarity = np.dot(query_embedding, context_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
            )
            
            # Check entity overlap
            query_entities = set(self.extract_fashion_entities(query).items())
            context_entities = set(self.extract_fashion_entities(context_text).items())
            entity_overlap = len(query_entities.intersection(context_entities))
            
            # Consider context relevant if either similarity is high or entities overlap
            if similarity > relevance_threshold or entity_overlap > 0:
                relevant_contexts += 1
        
        # Return True if at least 2 relevant contexts found
        return relevant_contexts >= 2

    def _generate_without_context(self, query: str) -> str:
        """Generate response when no relevant context is available"""
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Create a more focused prompt for contextless generation
        prompt = f"""As a fashion expert, provide specific advice about this clothing choice:
        Q: {processed_query}
        A: Based on fashion principles, """
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,  # Slightly lower temperature for more focused responses
                top_p=0.9,
                no_repeat_ngram_size=3,
                repetition_penalty=1.8
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._clean_response(response)

    def _calculate_confidence(self, response: str, contexts: List[Dict], has_relevant_context: bool) -> float:
        """Calculate confidence score for the generated response"""
        if not response or response.startswith("I apologize"):
            return 0.0
        
        confidence_scores = []
        
        # Base response quality score
        response_quality = self._get_response_quality_score(response)
        confidence_scores.append(response_quality)
        
        # Context-based confidence
        if has_relevant_context:
            context_score = self._get_context_similarity_score(response, contexts)
            confidence_scores.append(context_score * 1.2)  # Weight context score higher
        else:
            # Adjust confidence for contextless responses
            contextless_score = self._get_contextless_confidence(response)
            confidence_scores.append(contextless_score)
        
        # Entity coverage score
        entity_score = self._get_entity_coverage_score(response)
        confidence_scores.append(entity_score)
        
        # Adjust final confidence based on context availability
        final_confidence = sum(confidence_scores) / len(confidence_scores)
        if not has_relevant_context:
            final_confidence *= 0.9  # Slightly reduce confidence for contextless responses
        
        return min(final_confidence, 1.0)

    def _get_contextless_confidence(self, response: str) -> float:
        """Calculate confidence for responses generated without context"""
        score = 0.0
        
        # Check response structure
        doc = self.nlp(response)
        
        # Check for fashion-specific reasoning
        fashion_reasoning_terms = {
            "because", "since", "as", "therefore", "considering",
            "based on", "due to", "given that"
        }
        if any(term in response.lower() for term in fashion_reasoning_terms):
            score += 0.3
        
        # Check for specific recommendations
        if any(token.pos_ == "VERB" for token in doc):
            score += 0.2
        
        # Check for fashion terminology
        fashion_terms = {
            "style", "fashion", "wear", "outfit", "look", "trend",
            "season", "color", "pattern", "fabric", "material"
        }
        term_count = sum(1 for term in fashion_terms if term in response.lower())
        score += min(0.1 * term_count, 0.3)
        
        # Check for balanced response length
        words = len(response.split())
        if 20 <= words <= 50:
            score += 0.2
        elif 10 <= words < 20:
            score += 0.1
        
        return min(score, 1.0)

    def _prepare_context(self, contexts: List[Dict]) -> str:
        """Prepare context string from retrieved results"""
        context_text = "Previous relevant conversations:\n"
        for ctx in contexts[:3]:  # Use top 3 contexts
            context_text += f"Q: {ctx['data']['query']}\nA: {ctx['data']['response']}\n\n"
        return context_text

    def _generate_with_context(self, query: str, context: str) -> str:
        """Generate response using the model with context"""
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Preprocess context
        processed_context = self._preprocess_text(context)
        
        prompt = f"""{processed_context}Fashion advice for specific clothing items:
        Q: {processed_query}
        A: Let me give specific advice about this clothing choice."""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
                repetition_penalty=1.8
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._clean_response(response)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better generation quality"""
        # Use spaCy for preprocessing
        doc = self.nlp(text)
        
        # Basic preprocessing steps
        processed_text = text.lower()  # Convert to lowercase
        
        # Remove extra whitespace
        processed_text = " ".join(processed_text.split())
        
        # Remove special characters but keep essential punctuation
        processed_text = "".join(c for c in processed_text if c.isalnum() or c in " .,!?")
        
        # Optional: Extract key fashion-related terms using spaCy entities
        fashion_terms = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "COLOR", "MATERIAL"]]
        
        # Ensure fashion terms are preserved in the processed text
        for term in fashion_terms:
            if term.lower() not in processed_text:
                processed_text += f" {term.lower()}"
                
        return processed_text

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Extract only the response part after the last "A:"
        response_parts = response.split("A:")
        if len(response_parts) > 1:
            response = response_parts[-1].strip()
            response = response.split('\n')[0].split('|')[0].strip()
            if "Let me give specific advice" in response:
                response = response.replace("Let me give specific advice about this clothing choice.", "").strip()
        return response

    def _get_conversation_by_id(self, idx: int) -> Optional[Dict]:
        """Retrieve conversation from database by ID"""
        self.cursor.execute('SELECT * FROM conversations WHERE id = ?', (idx,))
        result = self.cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'query': result[1],
                'response': result[2],
                'timestamp': result[4],
                'feedback_score': result[5]
            }
        return None

    def cleanup(self):
        """Cleanup resources"""
        self.conn.close()

    def get_text_embedding(self, text):
        """Convert text into a fixed-size embedding using SentenceTransformer."""
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(self.vector_dim).astype('float32')

# Example usage
if __name__ == "__main__":
    processor = EnhancedNLPProcessor()
    
    test_queries = [
        "I am wearing a wool dress in summer, is it okay?",
        "Can I wear a leather jacket in winter?",
        "Are white pants good for summer?"
    ]
    
    for query in test_queries:
        response, confidence, has_relevant_context = processor.generate_enhanced_response(query)
        print(f"\nQuery: {query}")
        print(f"Response (confidence: {confidence:.2f}, relevant context: {has_relevant_context}): {response}")
    
    processor.cleanup() 