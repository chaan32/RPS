import os
import google.generativeai as genai

from .prompts import SYSTEM_PROMPT, build_user_message, strip_code_fence


# gemini -> none use
def summarize_logs_to_html(date_str: str, logs: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(build_user_message(date_str, logs))
    return strip_code_fence(response.text)
