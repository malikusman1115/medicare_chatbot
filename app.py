import os
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
from state_handler import MedicareHandler

# Configure the logger
log_directory = 'conversation_logs'
os.makedirs(log_directory, exist_ok=True)

log_file = os.path.join(log_directory, 'conversation_log.log')

logger = logging.getLogger("ChatbotLogger")
logger.setLevel(logging.INFO)

# Set up a rotating log handler (5MB per file, with up to 5 backups)
handler = RotatingFileHandler(log_file, maxBytes=15 * 1024 * 1024, backupCount=12)
formatter = logging.Formatter('%(asctime)s - %(message)s')  
handler.setFormatter(formatter)
logger.addHandler(handler)

# ThreadPoolExecutor for async logging (system-managed worker count)
executor = ThreadPoolExecutor()  # No need to specify max_workers

class SessionManager:
    def __init__(self):
        self.conversation_states = {}
        self.last_interaction = {}
        self.empty_input_counts = {}
        self.session_counters = {}

    def initialize_session(self, session_id, handler):
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = handler.reset_state(session_id)
            self.empty_input_counts[session_id] = 0
            self.session_counters[session_id] = 0
            return True  # New session initialized
        return False  # Existing session

    def update_session(self, session_id, handler, user_input):
        self.session_counters[session_id] += 1
        response, intent, state = handler.generate_response(session_id, user_input)
        self.last_interaction[session_id] = time.time()
        return response, intent, state

    def log_conversation(self, session_id, message):
        log_entry = f"Session ID: {session_id} - Message: {message}"
        self.last_interaction[session_id] = time.time()
        # Log asynchronously
        executor.submit(self.async_log, log_entry)

    def async_log(self, log_entry):
        logger.info(log_entry)

    def cleanup_inactive_sessions(self, timeout=150):
        current_time = time.time()
        inactive_sessions = [sid for sid, last_time in self.last_interaction.items() if current_time - last_time > timeout]
        for sid in inactive_sessions:
            goodbye_message = "inactive removed."
            self.log_conversation(sid, f"Chatbot: {goodbye_message}")
            self.remove_session(sid)

    def remove_session(self, session_id):
        self.conversation_states.pop(session_id, None)
        self.last_interaction.pop(session_id, None)
        self.empty_input_counts.pop(session_id, None)
        self.session_counters.pop(session_id, None)


app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes
handler = MedicareHandler()
session_manager = SessionManager()  # Instantiate the SessionManager

@app.route('/', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input_text', '')
    session_id = data.get('session_id', '')

    if not session_id:
        return jsonify({"error": "Session ID is missing."}), 400

    # Initialize or update session
    new_session = session_manager.initialize_session(session_id, handler)

    # Process the user's input and generate a response
    response1, intent, state = session_manager.update_session(session_id, handler, user_input)

    # Log conversation and intent
    session_manager.log_conversation(session_id, f"User: {user_input}")
    session_manager.log_conversation(session_id, f"Chatbot: {response1}")
    session_manager.log_conversation(session_id, f"Intent: {intent}")

    # Send the response back to the user
    response_data = {"response": response1, "intent": intent, "state": state}
    return jsonify(response_data)


def monitor_sessions():
    while True:
        session_manager.cleanup_inactive_sessions()
        time.sleep(20)  # Check every 20 seconds


if __name__ == '__main__':
    monitor_thread = Thread(target=monitor_sessions, daemon=True)
    monitor_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
