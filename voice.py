# ============================================================
# voice.py - Voice Input & Output Features
# ============================================================
#
# PURPOSE:
#   This file handles all voice-related features:
#   1. Speech-to-Text (STT): Microphone → Text for input
#   2. Text-to-Speech (TTS): AI response → Audio for output
#
# LIBRARIES USED:
#   - SpeechRecognition: Records from microphone and sends to
#     Google's free speech-to-text API for transcription
#   - gTTS (Google Text-to-Speech): Converts text to MP3 audio
#     using Google's TTS engine (requires internet)
#
# NOTE FOR BEGINNERS:
#   Voice features require additional system libraries:
#   - For microphone: PyAudio (see installation notes below)
#   - For audio playback: pygame or playsound
#
# INSTALLATION NOTES:
#   On Ubuntu/Debian:  sudo apt-get install python3-pyaudio portaudio19-dev
#   On macOS:          brew install portaudio && pip install pyaudio
#   On Windows:        pip install pyaudio (usually works directly)
# ============================================================

import os
import io
import tempfile
import time
from typing import Optional, Tuple

# Text-to-Speech using Google's free TTS API
# Converts text to MP3 audio file
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("  ⚠️  gTTS not installed. Text-to-speech disabled.")
    print("     Install with: pip install gTTS")

# Speech Recognition - records audio and transcribes it
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("  ⚠️  SpeechRecognition not installed. Speech-to-text disabled.")
    print("     Install with: pip install SpeechRecognition pyaudio")


# ============================================================
# TEXT-TO-SPEECH FUNCTIONS
# ============================================================

def text_to_speech(text: str, language: str = "en") -> Optional[bytes]:
    """
    Convert text to speech audio bytes using Google TTS.
    
    HOW gTTS WORKS:
    1. Send text to Google's TTS API (free, no key needed)
    2. Google returns MP3 audio data
    3. We return the audio bytes for playback in Streamlit
    
    Args:
        text: The text to convert to speech
        language: Language code (e.g., "en" for English, "hi" for Hindi)
        
    Returns:
        MP3 audio bytes, or None if failed
    """
    if not GTTS_AVAILABLE:
        print("  ✗ gTTS not available for text-to-speech")
        return None

    # Limit text length to avoid very long audio
    # (gTTS handles long texts but it's better to chunk them)
    if len(text) > 5000:
        text = text[:5000] + "... Response truncated for audio."
        print("  ⚠️  Text truncated to 5000 chars for TTS")

    try:
        # Create gTTS object
        # slow=False = normal speaking speed (True = slower, clearer)
        tts = gTTS(text=text, lang=language, slow=False)

        # Save to an in-memory bytes buffer instead of a file
        # This avoids creating temporary files on disk
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Reset buffer position to start

        # Return the raw bytes
        audio_bytes = audio_buffer.read()
        print(f"  ✓ TTS generated: {len(audio_bytes)} bytes of audio")
        return audio_bytes

    except Exception as e:
        print(f"  ✗ TTS Error: {e}")
        return None


def text_to_speech_file(text: str, output_path: str = "response.mp3", language: str = "en") -> Optional[str]:
    """
    Convert text to speech and save as MP3 file.
    
    Args:
        text: Text to convert
        output_path: Where to save the MP3 file
        language: Language code
        
    Returns:
        Path to the created MP3 file, or None if failed
    """
    if not GTTS_AVAILABLE:
        return None

    try:
        tts = gTTS(text=text[:5000], lang=language, slow=False)
        tts.save(output_path)
        print(f"  ✓ Audio saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"  ✗ TTS file save error: {e}")
        return None


# ============================================================
# SPEECH-TO-TEXT FUNCTIONS
# ============================================================

def speech_to_text_from_microphone(
    timeout: int = 5,
    phrase_timeout: int = 3
) -> Tuple[Optional[str], str]:
    """
    Record from microphone and transcribe to text.
    
    HOW SpeechRecognition WORKS:
    1. Open microphone using PyAudio
    2. Calibrate for ambient noise (2 seconds)
    3. Listen for speech (stop when silence detected)
    4. Send audio to Google's free Speech-to-Text API
    5. Return transcribed text
    
    Args:
        timeout: Max seconds to wait for speech to begin
        phrase_timeout: Max seconds of silence before stopping
        
    Returns:
        Tuple of (transcribed_text_or_None, status_message)
    """
    if not SR_AVAILABLE:
        return None, "SpeechRecognition library not available. Install: pip install SpeechRecognition pyaudio"

    # Create a recognizer instance
    # The recognizer handles all the audio processing
    recognizer = sr.Recognizer()

    try:
        # Open the default microphone as the audio source
        with sr.Microphone() as source:
            print("  🎤 Adjusting for ambient noise...")

            # Calibrate the recognizer to the ambient noise level
            # This helps filter out background noise
            # duration=1 means listen for 1 second to learn background noise
            recognizer.adjust_for_ambient_noise(source, duration=1)

            print("  🎤 Listening... (speak now)")

            # Listen for speech
            # timeout: how long to wait for speech to start
            # phrase_time_limit: max length of recording
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_timeout
            )

            print("  🔄 Transcribing...")

            # Use Google's free speech recognition API
            # No API key needed for basic usage!
            text = recognizer.recognize_google(audio, language="en-US")

            print(f"  ✓ Transcribed: '{text}'")
            return text, f"Transcribed: '{text}'"

    except sr.WaitTimeoutError:
        # User didn't speak within the timeout period
        return None, "No speech detected. Please try again and speak clearly."

    except sr.UnknownValueError:
        # Speech was detected but couldn't be understood
        return None, "Could not understand the audio. Please speak clearly."

    except sr.RequestError as e:
        # Google's API wasn't accessible (network issue, etc.)
        return None, f"Speech recognition service error: {e}"

    except Exception as e:
        # Catch-all for other errors (often PyAudio/microphone issues)
        return None, f"Microphone error: {e}\nMake sure PyAudio is installed and microphone is connected."


def speech_to_text_from_audio_file(audio_file_path: str) -> Tuple[Optional[str], str]:
    """
    Transcribe speech from an audio file (WAV, MP3, etc.)
    
    Useful when the user uploads an audio recording instead of
    using the microphone directly.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Tuple of (transcribed_text_or_None, status_message)
    """
    if not SR_AVAILABLE:
        return None, "SpeechRecognition not available"

    recognizer = sr.Recognizer()

    try:
        # Load the audio file as a speech recognition audio source
        with sr.AudioFile(audio_file_path) as source:
            # record() reads the entire audio file
            audio = recognizer.record(source)

        # Transcribe using Google's API
        text = recognizer.recognize_google(audio)
        return text, f"Transcribed successfully: '{text[:100]}...'"

    except sr.UnknownValueError:
        return None, "Could not understand the audio file content."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"Audio file error: {e}"


def speech_to_text_from_uploaded_file(uploaded_audio) -> Tuple[Optional[str], str]:
    """
    Transcribe speech from a Streamlit uploaded audio file.
    
    When users upload audio through Streamlit's file_uploader,
    we need to save it temporarily before processing.
    
    Args:
        uploaded_audio: Streamlit UploadedFile object
        
    Returns:
        Tuple of (transcribed_text_or_None, status_message)
    """
    if not SR_AVAILABLE:
        return None, "SpeechRecognition not available"

    # Determine file extension from the uploaded file name
    suffix = "." + uploaded_audio.name.split(".")[-1]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_path = tmp_file.name

    try:
        # Transcribe the temporary file
        result, message = speech_to_text_from_audio_file(tmp_path)
        return result, message
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_voice_availability() -> dict:
    """
    Check which voice features are available.
    
    Returns:
        Dictionary showing status of each voice feature
    """
    status = {
        "tts_available": GTTS_AVAILABLE,
        "stt_available": SR_AVAILABLE,
        "microphone_available": False
    }

    # Check if microphone is actually accessible
    if SR_AVAILABLE:
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            # Try to find a microphone input device
            mic_count = 0
            for i in range(audio.get_device_count()):
                dev_info = audio.get_device_info_by_index(i)
                if dev_info.get("maxInputChannels", 0) > 0:
                    mic_count += 1
            audio.terminate()
            status["microphone_available"] = mic_count > 0
            status["mic_device_count"] = mic_count
        except Exception as e:
            status["microphone_error"] = str(e)

    return status


def get_supported_languages() -> dict:
    """
    Return supported TTS languages.
    
    Returns:
        Dictionary of language_code: language_name pairs
    """
    return {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "zh": "Chinese (Mandarin)",
        "ar": "Arabic",
        "pt": "Portuguese",
        "ru": "Russian"
    }
