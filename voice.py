"""
voice.py - Voice Input & Output Features (TTS + STT with Whisper)
"""

import io
import os
import tempfile
from typing import Optional, Tuple

# ── TTS ───────────────────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# ── Google STT (fallback) ─────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ── Whisper STT ───────────────────────────────────────────────────────────────
try:
    import whisper as _whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ── PyAudio (microphone) ──────────────────────────────────────────────────────
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# Cache loaded Whisper model in memory (loaded once per session)
_whisper_model = None


def _get_whisper_model(model_size: str = "base"):
    """Lazy-load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        _whisper_model = _whisper.load_model(model_size)
    return _whisper_model


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT-TO-SPEECH
# ══════════════════════════════════════════════════════════════════════════════

def text_to_speech(text: str, language: str = "en") -> Optional[bytes]:
    """Convert text to MP3 audio bytes using Google TTS."""
    if not GTTS_AVAILABLE:
        return None
    if not text or not text.strip():
        return None
    # Truncate for API limits
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


def text_to_speech_file(text: str, output_path: str = "response.mp3",
                        language: str = "en") -> Optional[str]:
    """Convert text to speech and save as an MP3 file."""
    if not GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text[:5000], lang=language, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS file save error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SPEECH-TO-TEXT — Whisper (primary)
# ══════════════════════════════════════════════════════════════════════════════

def speech_to_text_whisper(
    timeout: int = 5,
    phrase_timeout: int = 8,
    model_size: str = "base",
    language: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Record audio from the microphone and transcribe using OpenAI Whisper.

    Args:
        timeout: seconds to wait for speech to start
        phrase_timeout: max seconds of recording
        model_size: whisper model ('tiny', 'base', 'small', 'medium', 'large')
        language: optional ISO-639 code (e.g. 'en', 'hi'). None = auto-detect.

    Returns:
        (transcribed_text | None, status_message)
    """
    if not WHISPER_AVAILABLE:
        return None, (
            "Whisper is not installed. Run: pip install openai-whisper"
        )
    if not SR_AVAILABLE:
        return None, (
            "SpeechRecognition is required for microphone access. "
            "Run: pip install SpeechRecognition pyaudio"
        )

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_timeout
            )
    except sr.WaitTimeoutError:
        return None, "No speech detected. Please try again."
    except Exception as e:
        return None, f"Microphone error: {e}"

    # Write raw audio to a temp WAV file for Whisper
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(audio.get_wav_data())
            tmp_wav_path = tmp_wav.name

        model = _get_whisper_model(model_size)
        if model is None:
            return None, f"Could not load Whisper '{model_size}' model."

        result = model.transcribe(
            tmp_wav_path,
            language=language,
            fp16=False,  # safer on CPU
        )
        text = result.get("text", "").strip()

        try:
            os.unlink(tmp_wav_path)
        except OSError:
            pass

        if text:
            return text, f"Heard: '{text}'"
        return None, "Whisper returned an empty transcript. Please speak more clearly."

    except Exception as e:
        return None, f"Whisper transcription error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  SPEECH-TO-TEXT — Google (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def speech_to_text_google(
    timeout: int = 5,
    phrase_timeout: int = 8,
) -> Tuple[Optional[str], str]:
    """Transcribe microphone audio via Google Speech Recognition (online)."""
    if not SR_AVAILABLE:
        return None, (
            "SpeechRecognition not available. "
            "Run: pip install SpeechRecognition pyaudio"
        )
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_timeout
            )
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


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT — auto-selects best available engine
# ══════════════════════════════════════════════════════════════════════════════

def speech_to_text_from_microphone(
    timeout: int = 5,
    phrase_timeout: int = 8,
    prefer_whisper: bool = True,
    whisper_model: str = "base",
    language: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Record from microphone and transcribe.

    Tries Whisper first (if available and prefer_whisper=True),
    then falls back to Google Speech Recognition.
    """
    if prefer_whisper and WHISPER_AVAILABLE and SR_AVAILABLE:
        return speech_to_text_whisper(
            timeout=timeout,
            phrase_timeout=phrase_timeout,
            model_size=whisper_model,
            language=language,
        )
    # Google fallback
    return speech_to_text_google(timeout=timeout, phrase_timeout=phrase_timeout)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSCRIBE FROM UPLOADED AUDIO FILE (bonus utility)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio_file(
    file_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Transcribe an existing audio file with Whisper.
    Useful if you let users upload voice recordings.
    """
    if not WHISPER_AVAILABLE:
        return None, "Whisper is not installed. Run: pip install openai-whisper"
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    try:
        model = _get_whisper_model(model_size)
        result = model.transcribe(file_path, language=language, fp16=False)
        text = result.get("text", "").strip()
        return (text, "Transcription complete.") if text else (None, "Empty transcript.")
    except Exception as e:
        return None, f"Transcription error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_voice_availability() -> dict:
    """Return a dict of which voice features are available."""
    return {
        "tts_available":     GTTS_AVAILABLE,
        "stt_google":        SR_AVAILABLE,
        "stt_whisper":       WHISPER_AVAILABLE,
        "microphone":        PYAUDIO_AVAILABLE,
        "active_stt_engine": "whisper" if WHISPER_AVAILABLE else (
                              "google"  if SR_AVAILABLE      else "none"),
    }


def get_supported_languages() -> dict:
    """Return language codes and their display names for TTS."""
    return {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ar": "Arabic",
        "pt": "Portuguese",
        "zh": "Chinese",
        "ko": "Korean",
        "it": "Italian",
        "ru": "Russian",
    }