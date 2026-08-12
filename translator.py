

import os                                  
import sys                                 
from anthropic import Anthropic            
from pydantic import BaseModel, Field      

MODEL = "claude-sonnet-5"  
MAX_HISTORY_TURNS = 6  


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


class Translation(BaseModel):
    source_lang: str = Field(description="'vi' or 'en' — detected source language")
    target_lang: str = Field(description="'vi' or 'en' — the language translated into")
    translation: str = Field(description="the translated text")
    notes: str = Field(description="brief note on tricky phrasing, or empty string")


def trim_history(history):
    """Keep only the last N turns so the context window doesn't grow forever."""
    max_entries = MAX_HISTORY_TURNS * 2  
    return history[-max_entries:]        


def translate(client, history, text):
    messages = history + [{"role": "user", "content": text}]  
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,           
        system=SYSTEM_PROMPT,      
        messages=messages,        
        output_format=Translation, 
    )
    return response.parsed_output  


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")  
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your-key")
        sys.exit(1)  

    client = Anthropic(api_key=api_key)  
    history = []                         

    print("VI <-> EN Translator — type 'quit' to exit\n")
    while True:                         
        text = input("> ").strip()       

        if text.lower() in ("quit", "exit"):
            break                        

        if not text:
            continue                     

        try:
            result = translate(client, history, text)  
        except Exception as e:
            print(f"[Error] {e}")        
            continue                    

        print(f"  [{result.source_lang} -> {result.target_lang}] {result.translation}")
        if result.notes:                 
            print(f"  note: {result.notes}")

        history.append({"role": "user", "content": text})                      
        history.append({"role": "assistant", "content": result.model_dump_json()})  
        history = trim_history(history)  


if __name__ == "__main__":  
    main()
