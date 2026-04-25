"""
voice.py - Voice Input & Output Features (TTS + STT)
"""

import os
import io
import tempfile
from typing import Optional, Tuple

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


def text_to_speech(text: str, language: str = "en") -> Optional[bytes]:
    """Convert text to MP3 audio bytes using Google TTS."""
    if not GTTS_AVAILABLE:
        return None
    if len(text) > 5000:
        text = text[:5000] + "... Response truncated for audio."
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def text_to_speech_file(text: str, output_path: str = "response.mp3", language: str = "en") -> Optional[str]:
    """Convert text to speech and save as MP3 file."""
    if not GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text[:5000], lang=language, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS file save error: {e}")
        return None


def speech_to_text_from_microphone(timeout: int = 5, phrase_timeout: int = 8) -> Tuple[Optional[str], str]:
    """Record from microphone and transcribe to text."""
    if not SR_AVAILABLE:
        return None, "SpeechRecognition not available. Install: pip install SpeechRecognition pyaudio"
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
        text = recognizer.recognize_google(audio, language="en-US")
        return text, f"Heard: '{text}'"
    except sr.WaitTimeoutError:
        return None, "No speech detected. Please try again."
    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please speak clearly."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"Microphone error: {e}"


def check_voice_availability() -> dict:
    """Check which voice features are available."""
    return {
        "tts_available": GTTS_AVAILABLE,
        "stt_available": SR_AVAILABLE,
    }


def get_supported_languages() -> dict:
    """Return supported TTS languages."""
    return {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ar": "Arabic",
        "pt": "Portuguese",
    }