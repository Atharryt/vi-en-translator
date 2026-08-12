"""
TRACK 0: system prompt design, structured reasoning, context assembly
TRACK 2: tool/function calling — schema design, call->execute->feed-back loop
TRACK 3: ReAct agent loop with scratchpad + persisted long-term memory + reflection
"""

import os
import sys
import json
from datetime import datetime
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError
import voyageai

import audio_tools
import glossary

MODEL = "claude-sonnet-5"
MEMORY_FILE = "memory.json"

TOOLS = [
    {
        "name": "transcribe_audio",
        "description": "Convert a speech audio file into text. Use this first on any raw audio file.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Path to the audio file"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "lookup_glossary",
        "description": "Search a glossary of tricky terms (jargon, idioms) by meaning. Use this if the transcribed text might contain a term that needs a specific correct translation.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The text to check against the glossary"}},
            "required": ["query"],
        },
    },
    {
        "name": "synthesize_speech",
        "description": "Speak text out loud. Use this as the LAST step, once you have the final translation.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to speak"}},
            "required": ["text"],
        },
    },
]

class TranscribeInput(BaseModel):
    file_path: str

class GlossaryInput(BaseModel):
    query: str

class SpeakInput(BaseModel):
    text: str

SYSTEM_PROMPT = """You are a Vietnamese<->English speech translation agent.

Given a path to an audio file, your job is to:
1. Transcribe it
2. Detect the language and decide if a glossary lookup would help translation accuracy
3. Produce an accurate, natural, idiomatic translation
4. Speak the translation out loud

You have tools to do each of these. Reason briefly about each step before
acting. Use lookup_glossary only when the text might contain jargon, an
idiom, or a name that could be mistranslated - not for simple sentences.
When you have spoken the final translation, respond with a final text
message starting with "FINAL_TRANSLATION:" followed by the translation,
and do not call any more tools.
"""


def execute_tool(name, tool_input, voyage_client, glossary_vectors):
    try:
        if name == "transcribe_audio":
            validated = TranscribeInput(**tool_input)
            return audio_tools.transcribe_audio_file(validated.file_path)

        elif name == "lookup_glossary":
            validated = GlossaryInput(**tool_input)
            hits = glossary.retrieve_glossary_hits(voyage_client, glossary_vectors, validated.query)
            return hits if hits else "no relevant glossary terms found"

        elif name == "synthesize_speech":
            validated = SpeakInput(**tool_input)
            return audio_tools.speak_text(validated.text)

        else:
            return f"unknown tool: {name}"

    except ValidationError as e:
        return f"invalid arguments, not executed: {e}"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def run_agent(audio_file_path, client, voyage_client, glossary_vectors):
    scratchpad = []
    messages = [{"role": "user", "content": f"Process the audio file at: {audio_file_path}"}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                scratchpad.append(("thought", block.text.strip()))
                print(f"  [thought] {block.text.strip()}")

        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if not tool_calls:
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return final_text, scratchpad

        messages.append({"role": "assistant", "content": response.content})

        tool_result_blocks = []
        for call in tool_calls:
            result = execute_tool(call.name, call.input, voyage_client, glossary_vectors)
            scratchpad.append(("action", call.name, call.input, result))
            print(f"  [action] {call.name}({call.input}) -> {result}")
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": str(result),
            })
        messages.append({"role": "user", "content": tool_result_blocks})


def reflect_on_translation(client, final_answer):
    critique = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="You are a translation quality critic. Review the given translation result for accuracy and natural phrasing. If it's good, respond with exactly 'APPROVED'. If not, respond with a corrected version.",
        messages=[{"role": "user", "content": final_answer}],
    )
    return "".join(b.text for b in critique.content if b.type == "text")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key or not voyage_key:
        print("Error: set both ANTHROPIC_API_KEY and VOYAGE_API_KEY")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python3 agent.py <path-to-audio-file>")
        sys.exit(1)
    audio_file_path = sys.argv[1]

    client = Anthropic(api_key=api_key)
    voyage_client = voyageai.Client(api_key=voyage_key)
    glossary_vectors = glossary.embed_glossary(voyage_client)

    memory = load_memory()

    print(f"\nProcessing: {audio_file_path}\n")
    final_answer, scratchpad = run_agent(audio_file_path, client, voyage_client, glossary_vectors)

    print(f"\nResult: {final_answer}")

    if not final_answer.strip():
        print("\n[Warning] The agent stopped without producing a final answer - skipping reflection.")
        critique = "N/A - no final answer produced"
    else:
        print("\nReflecting on translation quality...")
        critique = reflect_on_translation(client, final_answer)
        print(f"Critic: {critique}")

    memory.append({
        "timestamp": datetime.now().isoformat(),
        "audio_file": audio_file_path,
        "result": final_answer,
        "critic_review": critique,
    })
    save_memory(memory)


if __name__ == "__main__":
    main()