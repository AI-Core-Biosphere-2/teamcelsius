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

def generate_expert_response(expert, context, data_summary=None, model_choice="phi3"):
    """
    Generates an expert response using the local LLM.
    
    If a data summary is provided, it is included in the prompt so the expert can base its answer on the actual data.
    The prompt instructs the expert to provide a clear, comprehensive, and actionable analysis.
    """
    if data_summary:
        prompt = (
            f"You are {expert}, a seasoned expert in environmental sensor data analysis. "
            "You have access to the following data summary extracted from the current dataset. "
            "Based on this data and the context provided, please deliver a detailed, actionable, and specific forecast and analysis. "
            f"Data Summary: {data_summary}\n"
            f"Context: {context}\n\n"
            "Provide your expert analysis:"
        )
    else:
        prompt = (
            f"You are {expert}, a seasoned expert in environmental sensor data analysis. "
            "Based on the context provided, please deliver a detailed, actionable, and specific forecast and analysis. "
            f"Context: {context}\n\n"
            "Provide your expert analysis:"
        )
    response = generate_summary(prompt, model=model_choice)
    add_expert_message(expert, response)
    return response

def get_conversation_log():
    """Returns the conversation log."""
    return conversation_log
