from transformers import pipeline
from typing import Dict, Tuple

class IntentClassifier:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification")
        self.intents = [
            "fashion_advice",
            "product_search",
            "size_guide",
            "care_instructions",
            "general_question",
            "greeting",
            "farewell"
        ]

    def classify_intent(self, text: str) -> Tuple[str, float]:
        result = self.classifier(
            text,
            candidate_labels=self.intents,
            hypothesis_template="This is a {} request."
        )
        return result['labels'][0], result['scores'][0] 