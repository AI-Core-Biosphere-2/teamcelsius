# experts.py
import datetime
from ai_integration import generate_summary

# Global conversation log.
conversation_log = []

def add_expert_message(expert, message):
    """Adds a message from an expert to the conversation log."""
    conversation_log.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expert": expert,
        "message": message
    })

def generate_expert_response(expert, context):
    """
    Generates an expert response using the local LLM.
    The prompt instructs the model to provide a clear, comprehensive, and actionable analysis.
    Uses the 'phi3' model for faster, focused responses.
    """
    prompt = (
        f"You are {expert}, a seasoned expert in environmental sensor data analysis. "
        "Your role is to provide a clear, comprehensive, and actionable forecast based on the data summary below. "
        "Please ensure your response is complete and does not include placeholders. "
        f"Context: {context}\n\nProvide your detailed expert analysis:"
    )
    response = generate_summary(prompt, model="phi3")
    add_expert_message(expert, response)
    return response

def get_conversation_log():
    """Returns the conversation log."""
    return conversation_log
