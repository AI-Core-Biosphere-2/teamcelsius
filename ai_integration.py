# ai_integration.py
import subprocess

def generate_summary(simulation_text, model="phi3"):
    """
    Uses the local AI model (via Ollama) to generate a natural language summary.
    If the command fails, returns a fallback message.
    """
    prompt = simulation_text
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        summary = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        summary = "Expert analysis could not be generated at this time."
    return summary
