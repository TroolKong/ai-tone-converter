import os
from google import genai

# Replace with your actual API key from Google AI Studio
API_KEY = "YOUR_GEMINI_API_KEY"

# Initialize the Gemini client
client = genai.Client(api_key=API_KEY)

# Send a prompt to the model
prompt = "Explain what an AI Product Engineer does in two inspiring sentences."

print("Sending prompt to Gemini...")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt,
)

# Print the output from Gemini
print("\nGemini Response:")
print(response.text)