import pyttsx3
import threading


class TTS:
    def __init__(self, rate=150):
        self.engine = pyttsx3.init()
        try:
            self.engine.setProperty("rate", int(rate))
        except Exception:
            pass
        self.lock = threading.Lock()
        self._stopped = False

    def _speak_blocking(self, text: str):
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()

    def say(self, text: str):
        if self._stopped:
            return
        t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        t.start()

    def stop(self):
        self._stopped = True
        try:
            with self.lock:
                self.engine.stop()
        except Exception:
            pass
