import subprocess

def query_llm(prompt):
    command = ["ollama", "run", "llama2", prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

if __name__ == "__main__":
    user_prompt = "Can you tell me how to import csv files?" #enter user query here
    response = query_llm(user_prompt)
    print("LLM response:", response)
