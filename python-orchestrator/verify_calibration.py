import requests
import google.generativeai as genai
import json
from config import GEMINI_API_KEY, ENV_SERVER_URL

genai.configure(api_key=GEMINI_API_KEY)

CALIBRATION_PROMPT = """
You are a physics agent in a simulation.
Your goal is to MEASURE the size of your 'forward' step.

I will give you your starting position.
You must output a plan to move 'forward' by 1 step, and then tell me what you expect.
Then I will give you the result.
You must then output the MEASURED step size.

Format:
{
  "thought": "reasoning",
  "action": "forward",
  "steps": 1
}

After I give you the new X, you must reply:
{
  "thought": "reasoning",
  "measured_step_size": 0.123
}
"""

def test_calibration(granularity="fine", model_name="gemini-2.0-flash"):
    print(f"\n--- Testing Calibration ({granularity}) ---")
    model = genai.GenerativeModel(model_name)
    
    # 1. Reset
    resp = requests.post(f"{ENV_SERVER_URL}/reset", json={
        "task": "throw", "taskVersion": "v2", 
        "gravity": 9.81, "seed": 1004
    })
    start_x = resp.json()["observation"]["agentPosition"][0]
    
    # 2. Agent Turn 1
    prompt = f"{CALIBRATION_PROMPT}\n\nSTART_X: {start_x}\nGranularity: {granularity.upper()}"
    response = model.generate_content(prompt)
    print("Agent Step 1 Plan:", response.text)
    
    # 3. Execute
    scale = 1.0
    if granularity == "medium": scale = 0.5
    if granularity == "fine": scale = 0.25
    
    requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward", "durationScale": scale})
    
    # 4. Get New Pos
    resp = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "idle", "durationScale": 0.01})
    end_x = resp.json()["observation"]["agentPosition"][0]
    delta = end_x - start_x
    print(f"Actual Delta: {delta:.4f}")
    
    # 5. Agent Turn 2 (Calibration)
    prompt2 = f"You moved 1 step. Old X: {start_x}. New X: {end_x}. What is your step size?"
    chat = model.start_chat(history=[
        {"role": "user", "parts": [prompt]},
        {"role": "model", "parts": [response.text]}
    ])
    response2 = chat.send_message(prompt2)
    print("Agent Cali Result:", response2.text)
    
    return response2.text

if __name__ == "__main__":
    test_calibration("fine")
