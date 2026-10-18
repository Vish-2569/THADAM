import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly: Gemini connection successful."
    )

    print(response.text)

except Exception as error:
    print(f"Gemini test failed: {type(error).__name__}: {error}")