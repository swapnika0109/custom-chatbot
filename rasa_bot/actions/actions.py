from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_bot.nlp.enhanced_processor import EnhancedNLPProcessor
from rasa_bot.nlp.intent_classifier import IntentClassifier
from rasa_bot.nlp.processor import NLPProcessor
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

outfits = {
    'summer': {
        'casual': "light cotton t-shirt and shorts",
        'formal': "linen suit with a light shirt"
    },
    'winter': {
        'casual': "wool sweater and jeans",
        'formal': "wool suit with a turtleneck"
    }
    # Add more seasons and occasions as needed
}

class ActionRecommendOutfit(Action):
    def __init__(self):
        super().__init__()
        self.nlp_processor = EnhancedNLPProcessor()
        self.outfits = {
            'summer': {
                'casual': "light cotton t-shirt and shorts",
                'formal': "linen suit with a light shirt"
            },
            'winter': {
                'casual': "wool sweater and jeans",
                'formal': "wool suit with a turtleneck"
            }
            # Add more seasons and occasions as needed
        }

    def name(self) -> Text:
        return "action_recommend_outfit"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get the latest user message
        latest_message = tracker.latest_message.get('text', '')
        
        # Process with enhanced NLP
        entities = self.nlp_processor.extract_fashion_entities(latest_message)
        logger.debug('Entities: %s', entities)
        
        # Use existing slots with enhanced entity extraction
        season = tracker.get_slot('season') or (entities.get('seasons') and entities['seasons'][0])
        occasion = tracker.get_slot('occasion') or (entities.get('occasions') and entities['occasions'][0])
        
        # Try Rasa's predefined responses first
        if season and occasion:
            outfit = self.outfits.get(season, {}).get(occasion, "")
            if outfit:
                dispatcher.utter_message(f"For a {occasion} event in {season}, I recommend: {outfit}")
                return []
        
        # If Rasa doesn't have a suitable response, use enhanced processor
        enhanced_response, confidence, has_relevant_context = self.nlp_processor.generate_enhanced_response(latest_message)
        if enhanced_response and confidence > 0.7:  # Only use if confidence is high enough
            dispatcher.utter_message(enhanced_response)
            return []
            
        # Fall back to default response if neither method works
        dispatcher.utter_message("To give you the best outfit recommendation, could you tell me the season and occasion?")
        return []

class ActionGetFashionTrends(Action):
    def name(self) -> Text:
        return "action_get_fashion_trends"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        season = tracker.get_slot('season')
        
        trends = {
            'summer': [
                "Pastel colors",
                "Crop tops",
                "High-waisted shorts",
                "Floral prints",
                "Oversized sunglasses"
            ],
            'winter': [
                "Layered looks",
                "Chunky knits",
                "Leather boots",
                "Plaid patterns",
                "Faux fur accessories"
            ]
        }
        
        if season:
            season_trends = trends.get(season, [])
            if season_trends:
                trend_text = "\n- ".join(season_trends)
                dispatcher.utter_message(f"Current {season} trends include:\n- {trend_text}")
            else:
                dispatcher.utter_message("I can tell you about current trends! Which season are you interested in?")
        else:
            dispatcher.utter_message("I'd love to share fashion trends with you! Which season would you like to know about?")
        
        return []

class ActionStyleAdvice(Action):
    def name(self) -> Text:
        return "action_style_advice"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        outfit = tracker.get_slot('outfit')
        
        style_tips = {
            'dress': [
                "Choose accessories that complement the dress style",
                "Consider the event dress code",
                "Pay attention to the length and fit",
                "Select appropriate shoes for the occasion"
            ],
            'suit': [
                "Ensure proper tailoring",
                "Match belt with shoes",
                "Choose a tie that complements the suit color",
                "Consider the occasion for style choices"
            ]
        }
        
        if outfit:
            tips = style_tips.get(outfit, [])
            if tips:
                tips_text = "\n- ".join(tips)
                dispatcher.utter_message(f"Here are some styling tips for your {outfit}:\n- {tips_text}")
            else:
                dispatcher.utter_message("I can provide styling advice! What specific piece would you like tips for?")
        else:
            dispatcher.utter_message("I'd be happy to give you styling advice! What piece of clothing would you like to know about?")
        
        return []

class ActionLocalFashion(Action):
    def name(self) -> Text:
        return "action_local_fashion"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        country = tracker.get_slot('country')
        
        fashion_styles = {
            'France': "Known for haute couture, elegant minimalism, and chic casual wear",
            'Japan': "Combines traditional elements with avant-garde street fashion",
            'Italy': "Emphasizes luxury materials, tailoring, and bold accessories",
            'USA': "Diverse styles ranging from preppy to streetwear, with focus on comfort"
        }
        
        if country:
            style = fashion_styles.get(country)
            if style:
                dispatcher.utter_message(f"Fashion in {country}: {style}")
            else:
                dispatcher.utter_message("I can tell you about fashion in different countries! Which country's style interests you?")
        else:
            dispatcher.utter_message("I'd love to share information about local fashion! Which country would you like to know about?")
        
        return []

class ActionClassifyIntent(Action):
    def __init__(self):
        super().__init__()
        self.classifier = IntentClassifier()
        self.nlp_processor = EnhancedNLPProcessor()
    
    def name(self) -> Text:
        return "action_classify_intent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '')
        
        # Get intent classification
        intent, confidence = self.classifier.classify_intent(user_message)
        
        # Get enhanced response based on intent
        response, response_confidence, has_context = self.nlp_processor.generate_enhanced_response(user_message)
        
        # Combine both results for a more informed response
        if confidence > 0.7:  # High confidence in intent classification
            dispatcher.utter_message(
                text=response,
                json_message={
                    "detected_intent": intent,
                    "confidence": confidence,
                    "has_relevant_context": has_context
                }
            )
        else:
            # Fall back to general response with lower confidence
            dispatcher.utter_message(
                text="I'm not entirely sure what you're asking about. Could you please rephrase your question?",
                json_message={
                    "detected_intent": intent,
                    "confidence": confidence
                }
            )
        
        return [] 