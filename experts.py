# experts.py
from ai_integration import generate_summary
import datetime

# Global conversation log list; each entry is a dict with timestamp, expert, and message.
conversation_log = []

def add_expert_message(expert, message):
    """
    Adds a message from an expert to the conversation log.
    """
    conversation_log.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expert": expert,
        "message": message
    })

def generate_expert_response(expert, context):
    """
    Generates an expert response using the local AI model.
    expert: role name (e.g., "Temperature Expert")
    context: string context from prior discussion or data summary.
    Returns the expert's response.
    """
    prompt = (f"You are {expert}. Given the following discussion context: {context} "
              "provide your analysis and forecast for your variable.")
    response = generate_summary(prompt)
    add_expert_message(expert, response)
    return response

def get_conversation_log():
    """
    Returns the conversation log.
    """
    return conversation_log
