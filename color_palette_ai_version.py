
import google.generativeai as genai

genai.configure(api_key="AIzaSyAU-19IxU57eamhjBp2jFhSARoMRfOklUo")

# Create the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "max_output_tokens": 2048,
  "response_mime_type": "text/plain",
}
    
model = genai.GenerativeModel(
  model_name="gemini-1.0-pro",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

def generate_response(input_string: str) -> str:
    """Generates a response from the AI model based on the input string."""
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(input_string)
    return response.text
