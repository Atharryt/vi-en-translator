'EOF'
"""
Vietnamese <-> English Translator — Track 0 build.
Demonstrates: messages API, system/user prompt design, structured JSON
output, temperature control, and token-budget-aware conversation history.
"""

import os                                  # lets us read the API key from the terminal environment
import sys                                 # lets us stop the program cleanly with an error code
from anthropic import Anthropic            # the official Claude SDK — this is what talks to the API
from pydantic import BaseModel, Field      # used to define the exact shape Claude's reply must match

MODEL = "claude-sonnet-5"  # which Claude model to call — one place to change if we swap models later
MAX_HISTORY_TURNS = 6  # keep last N exchanges — real token budgeting, not infinite growth

# SYSTEM_PROMPT is Claude's permanent "job description" — sent on every call, never shown to the user.
# The two "Input/Output" examples inside it are few-shot examples: showing the model what a good
# answer looks like works better than just describing the task in words.
SYSTEM_PROMPT = """You are a professional Vietnamese-English interpreter.

Rules:
- Detect the source language automatically (Vietnamese or English).
- Translate naturally and idiomatically — never word-for-word.
- Preserve tone (casual vs formal) and keep names/numbers exact.
- If a phrase is idiomatic or hard to translate directly, add a short note.

Examples:
Input: "Cho toi xin mot ly ca phe sua da"
Output: source_lang=vi, translation="Can I get an iced milk coffee, please",
notes="'cho toi xin' is a polite request form, softened naturally in English"

Input: "Where's the nearest bus stop?"
Output: source_lang=en, translation="Tram xe buyt gan nhat o dau?", notes=""
"""

# This class is a schema, not a function — it forces Claude's reply into exactly these 4 fields,
# so we never have to guess-parse free text. Each Field's description is sent to Claude too.
class Translation(BaseModel):
    source_lang: str = Field(description="'vi' or 'en' — detected source language")
    target_lang: str = Field(description="'vi' or 'en' — the language translated into")
    translation: str = Field(description="the translated text")
    notes: str = Field(description="brief note on tricky phrasing, or empty string")


def trim_history(history):
    """Keep only the last N turns so the context window doesn't grow forever."""
    max_entries = MAX_HISTORY_TURNS * 2  # each turn = 1 user + 1 assistant entry
    return history[-max_entries:]        # Python slicing: keep only the last max_entries items


def translate(client, history, text):
    messages = history + [{"role": "user", "content": text}]  # past turns + the new thing user typed
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,           # hard ceiling on reply length, so it can't run away in size/cost
        system=SYSTEM_PROMPT,      # the instructions/rules block defined above
        messages=messages,         # the conversation itself
        output_format=Translation, # forces the reply to match our schema and returns it as a Python object
    )
    return response.parsed_output  # the already-validated Translation object — no manual parsing needed


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")  # read the key from the terminal, never hardcode it here
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your-key")
        sys.exit(1)  # 1 means "exited due to an error", as opposed to 0 for success

    client = Anthropic(api_key=api_key)  # one authenticated connection, reused for every call below
    history = []                         # starts empty, grows as the conversation continues

    print("VI <-> EN Translator — type 'quit' to exit\n")
    while True:                          # loop forever until the user explicitly quits
        text = input("> ").strip()       # .strip() removes stray leading/trailing spaces or newline

        if text.lower() in ("quit", "exit"):
            break                        # leaves the while loop, program ends normally

        if not text:
            continue                     # user hit Enter with nothing typed — skip back and ask again

        try:
            result = translate(client, history, text)  # call the function defined above
        except Exception as e:
            print(f"[Error] {e}")        # print what went wrong instead of crashing the whole program
            continue                     # go back and let the user try again

        print(f"  [{result.source_lang} -> {result.target_lang}] {result.translation}")
        if result.notes:                 # only print the notes line if Claude actually gave one
            print(f"  note: {result.notes}")

        history.append({"role": "user", "content": text})                       # save what the user said
        history.append({"role": "assistant", "content": result.model_dump_json()})  # save Claude's reply
        history = trim_history(history)  # immediately shrink history back down to the last N turns


if __name__ == "__main__":  # only run main() when this file is executed directly, not when imported
    main()
