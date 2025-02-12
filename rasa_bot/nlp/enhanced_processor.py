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

    def generate_enhanced_response(self, query: str) -> Tuple[str, float]:
        """Generate response with context and confidence score"""
        try:
            # Get relevant contexts
            contexts = self.hybrid_search(query)
            
            # Prepare prompt with contexts
            context_text = self._prepare_context(contexts)
            
            # Generate response
            response = self._generate_with_context(query, context_text)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(response, contexts)
            
            # Store conversation if confidence is high enough
            if confidence > 0.7:
                self._store_conversation(query, response)
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I couldn't generate a response.", 0.0

    def _prepare_context(self, contexts: List[Dict]) -> str:
        """Prepare context string from retrieved results"""
        context_text = "Previous relevant conversations:\n"
        for ctx in contexts[:3]:  # Use top 3 contexts
            context_text += f"Q: {ctx['data']['query']}\nA: {ctx['data']['response']}\n\n"
        return context_text

    def _generate_with_context(self, query: str, context: str) -> str:
        """Generate response using the model with context"""
        prompt = f"""{context}Fashion advice for specific clothing items:
        Q: {query}
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

    def _calculate_confidence(self, response: str, contexts: List[Dict]) -> float:
        """Calculate confidence score for the generated response"""
        # Implement your confidence calculation logic here
        # This is a simple example
        if not response or response.startswith("I apologize"):
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Add confidence based on context similarity
        if contexts:
            confidence += 0.3
            
        # Add confidence based on response length
        if len(response.split()) > 10:
            confidence += 0.2
            
        return min(confidence, 1.0)

    def _store_conversation(self, query: str, response: str):
        """Store conversation in database"""
        embedding = self.get_text_embedding(query)
        
        # Store in database
        self.cursor.execute('''
            INSERT INTO conversations (query, response, embedding, timestamp, version)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, response, embedding.tobytes(), datetime.now(), '1.0'))
        
        conversation_id = self.cursor.lastrowid
        
        # Store entities
        entities = self.extract_fashion_entities(query)
        for entity_type, values in entities.items():
            for value in values:
                self.cursor.execute('''
                    INSERT INTO entity_index (conversation_id, entity_type, entity_value)
                    VALUES (?, ?, ?)
                ''', (conversation_id, entity_type, value))
        
        self.conn.commit()
        
        # Update FAISS index
        self.index.add(np.array([embedding]))

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

# Example usage
if __name__ == "__main__":
    processor = EnhancedNLPProcessor()
    
    test_queries = [
        "I am wearing a wool dress in summer, is it okay?",
        "Can I wear a leather jacket in winter?",
        "Are white pants good for summer?"
    ]
    
    for query in test_queries:
        response, confidence = processor.generate_enhanced_response(query)
        print(f"\nQuery: {query}")
        print(f"Response (confidence: {confidence:.2f}): {response}")
    
    processor.cleanup() 