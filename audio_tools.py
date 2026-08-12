"""
Audio I/O — speech-to-text (ASR) and text-to-speech (TTS).
"""

import whisper
import pyttsx3

_whisper_model = None

def transcribe_audio_file(file_path: str) -> str:
    global _whisper_model
    if _whisper_model is None:
        print("  (loading speech-to-text model, first time only...)")
        _whisper_model = whisper.load_model("base")
    result = _whisper_model.transcribe(file_path)
    return result["text"].strip()


def speak_text(text: str) -> str:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return "spoken successfully"