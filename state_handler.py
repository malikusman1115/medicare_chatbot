import json
import pickle
import re
import numpy as np
import tensorflow as tf
from autocorrect import autocorrect
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from word2number import w2n
# Load responses from JSON
with open("responses.json", "r") as f:
    responses = json.load(f)
    number_patterns = responses["number_patterns"]
    intent_response_map = responses["intent_response_map"]

class MedicareHandler:
    def __init__(self):
        # Initialize model, tokenizer, and label encoder
        self.model = TFAlbertForSequenceClassification.from_pretrained('./weights/model18oct')
        self.tokenizer = AlbertTokenizer.from_pretrained('./weights/tokenizer18oct')
        label_encoder_path = './weights/encoder18oct.pkl'
        
        # Initialize state dictionaries
        self.state = {}  # Holds session states
        self.session_timers = {}  # Holds session timers

        # Load label encoder
        with open(label_encoder_path, 'rb') as file:
            self.label_encoder = pickle.load(file)

    def reset_state(self, session_id):
        """Initialize or reset the state for a given session"""
        self.state[session_id] = {
            "state": "greeting",
            "medicare_part_a_or_b": False,
            "last_message": "",
            "negative_responses_count": 0,
            "personal_query_count": 0,
            "question_count": 0,
            "card_count": 0,
            "transfer_general": 0,
            "greet_and_stop_count": 0,
            "positive_counter": 0,
            "bot_count": 0,
            "already_count": 0,
            "address_count": 0,
            "Name_count": 0,
            "location_count": 0,
            "response1": "",
            "intent1": "",
            "general_query_count": 0,
            "marketing_count": 0,
            "trust_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "back_count": 0,
            "start_count": 0,
            "empty_msg_count": 0,
            "empty_inputs": [],
            "intent_temp": 0,
            "non_empty_inputs": 0,
            "repeat_request_count":0,
            "global_intent_count":0,
            "medicare_card_1_seen":False,
            "global_empty_count":0,
            "conversation_history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "I'm from Medicare. This is Sophia calling. How are you today?"}
            ]
        }

    def reset_counters(self, session_id,decrement_counter=None):
        """Reset all counters in the state"""
        if session_id in self.state:
            state = self.state[session_id]
            counter_fields = [
                "negative_responses_count",
                "personal_query_count",
                "question_count",
                "card_count",
                "transfer_general",
                "greet_and_stop_count",
                "positive_counter",
                "bot_count",
                "already_count",
                "address_count",
                "Name_count",
                "location_count",
                "general_query_count",
                "marketing_count",
                "trust_count",
                "positive_count",
                "negative_count",
                "back_count",
                "start_count",
                "empty_msg_count",
                "intent_temp",
                "non_empty_inputs",
                "repeat_request_count",
                "intent_count",
                "global_intent_count",
                "medicare_card_1_seen"
            ]
            # Decrement a specific counter if specified
        if decrement_counter and decrement_counter in counter_fields:
            state[decrement_counter] = max(0, state.get(decrement_counter, 0) - 1)

        # Reset all other counters to zero
        for field in counter_fields:
            if field != decrement_counter:
                state[field] = 0
        
        state["empty_inputs"] = []
        state["medicare_card_1_seen"] = False
        

    def predict_intent(self, text):
        """Predict intent from text using the ALBERT model"""
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors="tf",
            max_length=50,
            padding=True,
            truncation=True
        )
        logits = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        predicted_class_id = np.argmax(probabilities, axis=-1)[0]
        return self.label_encoder.inverse_transform([predicted_class_id])[0], probabilities[0]

    def extract_age(self, sentence):
        """Extract age from sentence, handling both numeric and word formats"""
        # Check for numeric age
        digit_match = re.search(r'\b\d+\b', sentence)
        if digit_match:
            return int(digit_match.group())
        
        # Check for word format age
        words = re.sub(r'[^a-zA-Z\s-]', '', sentence).strip()
        try:
            age_in_number = w2n.word_to_num(words)
            return age_in_number
        except ValueError:
            return None
    
    def extract_consecutive_triplets(self,text):
        pattern = r'\b(?:' + '|'.join(number_patterns) + r')\b'
        words_to_numbers = lambda word: w2n.word_to_num(word) if not word.isdigit() else int(word)
        numbers, triplets = [], []
        for word in text.lower().split():
            try:
                num = words_to_numbers(word) if re.match(pattern, word) else None
                numbers.append(num) if num is not None else triplets.extend([(numbers[i], numbers[i+1], numbers[i+2]) 
                                for i in range(len(numbers)-2)]) or numbers.clear()
            except ValueError:
                pass
        if len(numbers) >= 3: triplets.extend([(numbers[i], numbers[i+1], numbers[i+2]) for i in range(len(numbers)-2)])
        return triplets if triplets else None
    

    def add_to_state(self, session_id, message, response=None):
        """Add message and response to conversation history"""
        if session_id not in self.state:
            self.reset_state(session_id)
        
        self.state[session_id]["last_message"] = message
        self.state[session_id]["conversation_history"].append({
            "role": "user",
            "content": message
        })
        
        if response:
            self.state[session_id]["conversation_history"].append({
                "role": "assistant",
                "content": response
            })

        
    def update_state_counters(self, state, intent, message):
        """
        Dynamically updates state counters and determines response keys based on intent.
        Returns tuple of (intent, response_key, should_exit)
        """
        
        counter_configs = {
            "empty": {
                "counter": "empty_msg_count",
                "response_key": lambda count: f"empty_{count}",
                "max_count": 2,
                "exit_intent": "DAIR"
            },
           
            "bot_query": {
                "counter": "bot_count",
                "response_key": lambda count: f"bot_query_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "negative_intro": {
                "counter": "negative_responses_count",
                "response_key": lambda count: f"negative_intro_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "personal_query": {
                "counter": "personal_query_count",
                "response_key": lambda count: f"personal_query_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "name_query": {
                "counter": "Name_count",
                "response_key": lambda count: f"name_query_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "trust_query": {
                "counter": "trust_count",
                "response_key": lambda count: f"trust_query_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "location_class": {
                "counter": "location_count",
                "response_key": lambda count: f"location_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "general_query": {
                "counter": "general_query_count",
                "response_key": lambda count: f"general_query_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "already_have": {
                "counter": "already_count",
                "response_key": lambda count: f"already_have_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "marketing": {
                "counter": "marketing_count",
                "response_key": lambda count: f"marketing_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
            "start_greeting": {
                "counter": "start_count",
                "response_key": lambda count: "start_greeting",
                "max_count": 2,
                "exit_intent": "NI"
            },
            "greet_back": {
                "counter": "start_count",
                "response_key": lambda count: "greet_back",
                "max_count": 2,
                "exit_intent": "NI"
            },
             "medicare_card": {
                "counter": "card_count",
                "response_key": lambda count: f"medicare_card_{count}",
                "max_count": 3,
                "exit_intent": "NI"
            },
             "positive_intro": {
                "counter": "positive_count",
                "response_key": lambda count: f"positive_intro_{count}",
                "max_count": 3,
                "exit_intent": "NI"

        },
        "repetition_query": {
                "counter": "repeat_request_count",
                "response_key": lambda count: "repetition",
                "max_count": 2,
                "exit_intent": "NI"

        }
        }

        if intent in counter_configs:
            config = counter_configs[intent]
            counter_name = config["counter"]

            # Update the specific counter for the intent
            current_count = state.get(counter_name, 0) + 1
            state[counter_name] = current_count

            # Update global_intent_count unless the intent is "empty"
            if intent != "empty":
                state["global_intent_count"] += 1

            response_key = config["response_key"](current_count)

            if current_count >= config["max_count"]:
                return config["exit_intent"], response_key, True
            return intent, response_key, False

        return intent, None, False
    def process_state(self, state, intent, message, state_responses):
        """Process the current state and return appropriate response"""
        default_response = state_responses.get("default", "I'm sorry, could you please repeat that?")
        # Define trust-related patterns
        trust_patterns = [
            r"(is it|are you).*(sure|real)",         # are you sure
            r"how.*trust",            # how can I trust
            r"(is|can|could).*trusted",  # can you be trusted
            r"(can|could).*confirm",  # can you confirm
            r"(is|can|could).*guarantee",# can you guarantee
            r"without a doubt",       # without a doubt
            r"(sure|confirm).*benefits",              # for sure
            r"can.*assure",           # can you assure
            r"confident.*benefits"    # confident about benefits
        ]
        lb_pattern = r'\b(?:english.*?(no|not)|no.*?english|spanish|french|english)\b'

        # Compile all sets of patterns
        compiled_patterns_trust = re.compile("|".join(trust_patterns), re.IGNORECASE)
        compiled_patterns_lb = re.compile(lb_pattern, re.IGNORECASE)
        # Function to check if the message matches name-related, trust-related, or interest-related patterns
        def check_patterns(message):
            if compiled_patterns_trust.search(message):
                return "trust_query"
            elif compiled_patterns_lb.search(message):
                return "LB"
            else:
                return None
        result = check_patterns(message)
        if result:
            intent = result
        
        ############# answering machine if three consecutive numbers #############3
        triplets = self.extract_consecutive_triplets(message)
        if triplets:
            intent = "A"
        ########### disconnected handle ######333
        if message == "disconnected":
            if state["intent1"] not in ["DNC","DNQ","NP","XFER","CALLBK","NI","A","LB","N"]:
                intent = "DAIR"
                response = state_responses.get("A", default_response)
            
            else:
                intent = state["intent1"]
        elif intent == "medicare_card":
                state["medicare_card_1_seen"] = True
        if state["medicare_card_1_seen"] and intent not in ["empty","repetition_query","XFER","DNQ","affirmative_class"] and message not in ["disconnected"]:
            intent = "medicare_card"
        new_intent, response_key, should_exit = self.update_state_counters(state, intent, message)
        intent = new_intent
        
        if should_exit:
            intent = new_intent
            response = state_responses.get("NI" if new_intent == "NI" else response_key, default_response)
            return response, intent
        
        if response_key:
            response = state_responses.get(response_key, default_response)
            return response, intent
        response = state_responses.get(intent, default_response)
        return response, intent
    
    def handle_repetition_query(self,state, state_responses, default_response):
            default_response = state_responses.get("default", "I'm sorry, could you please repeat that?")
            intent_key2 = state.get("intent1")
            print("intent key2 :",intent_key2)  # For debugging purposes
            # Fetch the response key based on the intent
            response_key2 = intent_response_map.get(intent_key2)
            if response_key2:
                print("Intent recognize d:", intent_key2)
                response = state_responses.get(response_key2, default_response)
            else:
                print("Intent not recognized:", intent_key2)
                response = state_responses.get(response_key2, default_response)
            return response
    
    
    def generate_response(self, session_id, message):
        """Main response generation method"""
        
        state = self.state.get(session_id)
        if not state:
            self.reset_state(session_id)
            state = self.state[session_id]
        

        message = message.lower()
        
        if state["state"] not in ["greeting", "ask_part_ab"]:
            message = autocorrect(message)
        
        if not message.strip():
            intent_temp = 1
        else:
            intent_temp = 0

        intent, probabilities = self.predict_intent(message)
        if intent == "XFER":
            intent = "XFER_1"
        if intent_temp:
            if intent_temp == 1:
                intent = "empty"
        else:
            intent = intent

        self.add_to_state(session_id, message)

        if intent == "age_ineligible":
            age = self.extract_age(message)
            if age is not None and age >= 65:
                intent = "positive_intro"
            else:
                intent = "DNQ"
       
        elif intent == "age_eligible":
            age = self.extract_age(message)
            if age is not None and age >= 65:
                intent = "positive_intro"
            elif age is not None and age <= 65:
                intent = "DNQ"
            else:
                intent = "DNQ"
        

        phrases = ["your phone number", "your email address", "your age"]
        pattern = r"|".join([re.escape(phrase) for phrase in phrases])
        if re.search(pattern, message):
            intent = "personal_query"

        # Get responses for current state
        state_responses = responses.get(state["state"], {})
        default_response =state_responses.get("default","I'm sorry, could you please repeat that?")
        
        # Process state and get response
        response, intent = self.process_state(state, intent, message, state_responses)
        self.add_to_state(session_id, message, response)
        # Update state based on current state and intent
        if state["state"] == "greeting":
            if intent == "DNQ":
                intent = "negative_intro"
                if state['intent1'] !="empty":
                    response =  state_responses.get("negative_intro_1")
                    state['state'] = "ask_part_ab"
            if state['intent1'] == "empty" and intent in ["affirmative_class","negative_intro","positive_intro","start_greeting","repetition_query"]:
                response =  state_responses.get("after_there")
                state["state"] = "ask_part_ab"
                self.reset_counters(session_id)
           
                self.reset_counters(session_id)
            elif intent not in ["DNQ", "DNC", "empty", "DAIR", "A"]:
                state["state"] = "ask_part_ab"
                self.reset_counters(session_id)
                state["global_empty_count"] = 0
            elif intent == "XFER_1":
                
                response = state_responses.get("XFER_1", "This is Sophia from Go Care Benefits., how have you been?")
                state["state"] = "ask_part_ab"
        elif state["state"] == "ask_part_ab":
            if intent == "empty": # if intent is empty and appear at different places in a state
                state["global_empty_count"]+=1
            if state["global_empty_count"]==3:
                intent = "NP"
                response = state_responses.get("A")

            if intent == "DNQ": 
                intent = "negative_intro"
                if state['intent1'] !="empty":
                    response =  state_responses.get("negative_intro_1")
                    state['state'] = "pitch"
            if state["intent1"] == "empty" and intent in ["affirmative_class","positive_intro","negative_intro","DNQ","start_greeting","repetition_query"] and not state["medicare_card_1_seen"] :
                    if intent == "DNQ": # empty and no medicard query hitted
                        intent = "negative_intro"
                    response = state_responses.get("after_there")
                    self.reset_counters(session_id) # reset counters to reset the empty_msg_count, other wise only two empty can appear in a state
            elif state['intent1'] == "empty" and intent in ["affirmative_class","negative_intro","positive_intro","start_greeting","repetition_query"]:
                if state["medicare_card_1_seen"]: #handle the medicare card scenario, if medicare card query hits, we have to see every intent as medicard
                    print("card seen")
                    if intent == "DNQ":
                        intent = "negative_intro"
                    response =  state_responses.get("after_there_card")
                    self.reset_counters(session_id) # reset empty_msg_count
            elif intent == "XFER_1":
                
                response = state_responses.get("XFER_1", "This is Sophia from Go Care Benefits., how have you been?")
                state["state"] ="pitch" 
            elif intent == "repetition_query":
                response = self.handle_repetition_query(state, state_responses, default_response)
            elif intent not in ["DNQ", "DNC", "empty", "DAIR", "A","NP"]: # do not change the state on these intents otherwise "disconnected" will not work accordingly
                state["state"] = "pitch" 
                self.reset_counters(session_id)
                state["global_empty_count"] = 0
            elif intent in ["DAIR", "NP"]: # overwrite DAIR into NP in ab state
                intent = "NP"
                response = state_responses.get("A")
            # elif intent == "XFER":
            #     intent = "trust_query_1"
            #     response = state_responses.get("trust_query_1", "Great. This will only take a moment of your time. I’m reaching out to share some exciting news about the updated Medicare benefits under Open Enrollment 2025 which includes dental, vision, prescription coverage; food cards and cash back as well. I believe you have active Medicare Part A or B, right?")
        elif state["state"] == "pitch":
            if intent in ["DAIR","N"]:
                intent = "NI"
                response = state_responses.get("A")
            if intent == "empty":
                state["global_empty_count"]+=1
            if state["global_empty_count"]==3:
                intent = "NI"
                response = state_responses.get("A")
            if state["intent1"] == "empty" and intent in ["affirmative_class","positive_intro","negative_intro","DNQ","start_greeting","repetition_query"] and not state["medicare_card_1_seen"] :
                    
                    if intent == "DNQ":
                        intent = "negative_intro"
                    response = state_responses.get("after_there")
                    self.reset_counters(session_id)
            elif state['intent1'] == "empty" and intent in ["affirmative_class","negative_intro","positive_intro","start_greeting","repetition_query"]:
                if state["medicare_card_1_seen"]:
                    
                    if intent == "DNQ":
                        intent = "negative_intro"
                    response =  state_responses.get("after_there_card")
                    self.reset_counters(session_id)
            elif state['intent1'] not in ["empty"] and intent in ["XFER_1", "affirmative_class"]:
                response = state_responses.get("affirmative_class")
                intent = "eligible"
                state["state"] = "transfer"
                self.reset_counters(session_id)
                state["global_empty_count"] = 0
            
            
            elif intent == "repetition_query":
                response = self.handle_repetition_query(state, state_responses, default_response)
        
        elif state["state"] == "transfer":
            
            if intent in ["positive_intro", "affirmative_class","XFER"]:
                response = state_responses.get("hold_on", " ")
                intent = "XFER"
            if intent=="DAIR":
                intent = "XFER"
            elif intent == "empty":
                intent = "XFER"
                response = state_responses.get("hold_on", " ")
            elif intent in ["negative_intro", "DNQ"]:
                intent = "negative_intro"
                state["negative_count"] += 1
                response_key = f"negative_intro_{state['negative_count']}"
                response = state_responses.get(response_key)
                if state["negative_count"] >= 2:
                    response = state_responses.get("N", "good bye")
                    intent = "NI"
            if message == "disconnected":
                intent = "NI"
                response = state_responses.get("A","good bye ")
            else:
                response = state_responses.get("default")
                intent = "XFER"
        
        # Update conversation history
        state["intent1"] = intent
        state["response1"] = response
        return response, intent, state["state"]