import pyttsx3
import threading


class TextToSpeech:
    """Offline text-to-speech using pyttsx3, runs in background thread."""

    def __init__(self, rate=150, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        self._speaking = False

    def speak(self, text):
        """Speak text in a non-blocking background thread."""
        if not text.strip() or self._speaking:
            return

        def _run():
            self._speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self._speaking = False

        threading.Thread(target=_run, daemon=True).start()

    def is_speaking(self):
        return self._speaking
