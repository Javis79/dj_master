import json
import math
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import sounddevice as sd
from mutagen import File as MutagenFile
from PySide6.QtCore import QSize, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QLinearGradient, QPainter, QPen, QPixmap, QBrush, QPainterPath, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
EQ_BANDS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

EQ_PRESETS = {
    "Flat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bass Boost": [7, 6, 4, 2, 0, 0, 1, 2, 2, 1],
    "Club / Electronic": [5, 4, 2, 0, -1, 2, 4, 5, 3, 2],
    "Rock / Alternative": [5, 4, -1, -2, -1, 1, 3, 4, 4, 3],
    "Hip-Hop / Rap": [6, 5, 2, 0, -1, -1, 1, 2, 3, 2],
    "Pop": [-2, -1, 2, 4, 4, 2, -1, -2, -1, 0],
    "Vocal": [-2, -1, 0, 2, 4, 5, 4, 2, 0, -1],
    "Acoustic": [2, 2, 1, 0, 0, 1, 2, 3, 2, 1],
    "Bright": [-2, -1, 0, 0, 1, 2, 4, 6, 7, 6],
}
TRANSITION_PRESETS = {
    "Equal Power Blend": {
        "duration": 16,
        "curve": "equal_power",
        "description": "Klasyczne plynne przejscie o stalej energii.",
    },
    "Smooth S-Curve": {
        "duration": 24,
        "curve": "s_curve",
        "description": "Dlugie, lagodne klubowe przejscie.",
    },
    "Bass Swap": {
        "duration": 12,
        "curve": "s_curve",
        "bass_dip": -5.0,
        "description": "Wycisza dol w srodku przejscia, zeby kicki sie nie gryzly.",
    },
    "Filter Fade": {
        "duration": 10,
        "curve": "s_curve",
        "high_dip": -4.0,
        "description": "Radiowe/praktyczne przejscie z delikatnym przygaszeniem gory.",
    },
    "Quick Cut": {
        "duration": 4,
        "curve": "sharp",
        "description": "Szybkie przejscie pod drop albo mocny punkt frazy.",
    },
    # --- NOWE PRZEJŚCIA ---
    "Gasolina Drop": {
        "duration": 2,
        "curve": "sharp",
        "bass_dip": 0.0,
        "description": "Agresywne i bardzo szybkie cięcie idealne pod dynamiczny drop lub reggaeton.",
    },
    "High-Pass Sweep": {
        "duration": 16,
        "curve": "equal_power",
        "bass_dip": -24.0, # Ekstremalne wycięcie basu w pierwszym utworze
        "description": "Płynne wejście z całkowitym wycięciem dołu (imitacja filtru High-Pass).",
    }
}


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20.0))


def decode_audio(path: Path) -> np.ndarray:
    command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "FFmpeg could not decode this audio file.") from exc

    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    usable = (len(samples) // 2) * 2
    if usable == 0:
        raise RuntimeError("FFmpeg decoded an empty audio stream.")
    return samples[:usable].reshape((-1, 2)).copy()


@dataclass
class Track:
    path: Path
    title: str
    artist: str
    duration: float
    cover_path: Path | None = None
    bpm: float = 0.0  # NOWE: pole na BPM

    @property
    def label(self) -> str:
        left = f"{self.artist} - {self.title}" if self.artist else self.title
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        bpm_str = f" [{int(self.bpm)} BPM]" if self.bpm > 0 else ""
        return f"{left}   {minutes}:{seconds:02d}{bpm_str}"

class DeckState:
    def __init__(self, name: str):
        self.name = name
        self.audio = np.zeros((0, 2), dtype=np.float32)
        self.position = 0.0  # Teraz float dla precyzyjnego ułamkowego skreczu!
        self.playing = False
        self.track: Track | None = None
        self.volume = 1.0
        self.spectrum = np.zeros(128, dtype=np.float32)
        
        # --- Zmienne dla systemu SCRATCH ---
        self.scratching = False
        self.scratch_target = 0.0

    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    def current_time(self) -> float:
        return float(self.position) / SAMPLE_RATE


class DJAudioEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.decks = [DeckState("A"), DeckState("B")]
        self.crossfader = 0.0
        self.master_db = 0.0
        self.eq_db = np.zeros(len(EQ_BANDS), dtype=np.float32)
        self._smoothed_eq_db = np.zeros(len(EQ_BANDS), dtype=np.float32)
        self._smoothed_master_db = 0.0
        self._filter_state = np.zeros((len(EQ_BANDS), 2, 2), dtype=np.float32)
        self.spectrum = np.zeros(128, dtype=np.float32)
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self.stream.start()
        # --- LAUNCHPAD: Pamięć i domyślne dźwięki ---
        self.sampler_pads = [None] * 8
        self.active_samples = [] 
        self._generate_default_samples()

    def close(self):
        self.stream.stop()
        self.stream.close()

    def load_deck(self, deck_index: int, track: Track, autoplay: bool = False):
        samples = decode_audio(track.path)
        
        # --- NORMALIZACJA GŁOŚNOŚCI (Wyrównywanie poziomów RMS) ---
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0.0001:
            target_rms = 0.15 # Nasz standardowy, przyjemny dla ucha punkt odniesienia
            gain = min(target_rms / rms, 4.0) # Maksymalnie 4-krotne wzmocnienie, by nie zepsuć szumu
            samples = samples * gain
            samples = np.clip(samples, -1.0, 1.0).astype(np.float32)

        with self.lock:
            deck = self.decks[deck_index]
            deck.audio = samples
            deck.position = 0.0
            deck.scratch_target = 0.0
            deck.scratching = False
            deck.track = track
            deck.playing = autoplay
            deck.spectrum.fill(0)

    def set_deck_playing(self, deck_index: int, value: bool):
        with self.lock:
            deck = self.decks[deck_index]
            if len(deck.audio):
                deck.playing = value

    def toggle_deck(self, deck_index: int):
        with self.lock:
            deck = self.decks[deck_index]
            if len(deck.audio):
                deck.playing = not deck.playing
                return deck.playing
        return False

    def seek_deck_ratio(self, deck_index: int, ratio: float):
        with self.lock:
            deck = self.decks[deck_index]
            if len(deck.audio):
                deck.position = float(np.clip(ratio, 0.0, 1.0) * (len(deck.audio) - 1))
                deck.scratch_target = deck.position # DODANO

    def set_deck_volume(self, deck_index: int, value: float):
        with self.lock:
            self.decks[deck_index].volume = float(np.clip(value, 0.0, 1.5))

    def set_crossfader(self, value: float):
        with self.lock:
            self.crossfader = float(np.clip(value, 0.0, 1.0))

    def deck_status(self, deck_index: int):
        with self.lock:
            deck = self.decks[deck_index]
            return {
                "track": deck.track,
                "playing": deck.playing,
                "current": deck.current_time(),
                "duration": deck.duration(),
                "spectrum": deck.spectrum.copy(),
            }

    def _generate_default_samples(self):
        # Programowe wygenerowanie domyślnych dźwięków bez zewnętrznych plików
        # Pad 1: Stopa (Kick)
        t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3), False).astype(np.float32)
        freq = np.geomspace(150, 30, len(t)).astype(np.float32)
        kick = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)).astype(np.float32)
        self.sampler_pads[0] = np.column_stack((kick, kick)) * 0.8
        
        # Pad 2: Hi-hat
        hh = ((np.random.rand(int(SAMPLE_RATE * 0.1)).astype(np.float32) * 2 - 1))
        hh = (hh * np.exp(-np.linspace(0, 30, len(hh)).astype(np.float32)))
        self.sampler_pads[1] = np.column_stack((hh, hh)) * 0.3
        
        # Pad 3: Laser / Synth Beep
        t_beep = np.linspace(0, 0.4, int(SAMPLE_RATE * 0.4), False).astype(np.float32)
        beep = (np.sin(2 * np.pi * 880 * t_beep) * np.exp(-t_beep * 5)).astype(np.float32)
        self.sampler_pads[2] = np.column_stack((beep, beep)) * 0.4

    def play_sample(self, index: int):
        with self.lock:
            if 0 <= index < 8 and self.sampler_pads[index] is not None:
                self.active_samples.append({"audio": self.sampler_pads[index], "pos": 0})
                
    def load_sample(self, index: int, path: Path):
        try:
            audio = decode_audio(path)
            with self.lock:
                self.sampler_pads[index] = audio
            return True
        except:
            return False

    @staticmethod
    def _peaking_coefficients(freq: float, gain_db: float, q: float = 1.15):
        a = 10 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * freq / SAMPLE_RATE
        alpha = math.sin(omega) / (2.0 * q)
        cos_omega = math.cos(omega)
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / a
        return (
            np.float32(b0 / a0),
            np.float32(b1 / a0),
            np.float32(b2 / a0),
            np.float32(a1 / a0),
            np.float32(a2 / a0),
        )

    def _apply_biquad(self, samples: np.ndarray, band_index: int, coeffs):
        b0, b1, b2, a1, a2 = coeffs
        out = np.empty_like(samples)
        for channel in range(2):
            z1, z2 = self._filter_state[band_index, channel]
            for i, x in enumerate(samples[:, channel]):
                y = (b0 * x) + z1
                z1 = (b1 * x) - (a1 * y) + z2
                z2 = (b2 * x) - (a2 * y)
                out[i, channel] = y
            self._filter_state[band_index, channel, 0] = z1
            self._filter_state[band_index, channel, 1] = z2
        return out

    def _update_spectrum(self, target: np.ndarray, samples: np.ndarray):
        if len(samples) < BLOCK_SIZE:
            block = np.zeros(BLOCK_SIZE, dtype=np.float32)
            block[: len(samples)] = np.mean(samples, axis=1)
        else:
            block = np.mean(samples[:BLOCK_SIZE], axis=1)
        mono = np.abs(np.fft.rfft(block * np.hanning(BLOCK_SIZE).astype(np.float32)))
        target[:] = np.interp(
            np.linspace(0, len(mono) - 1, 128),
            np.arange(len(mono)),
            np.clip(np.log10(mono + 1e-6) + 5.0, 0.0, 5.0) / 5.0,
        ).astype(np.float32)

    def _master_process(self, samples: np.ndarray) -> np.ndarray:
        out = samples.astype(np.float32, copy=True)
        self._smoothed_eq_db += (self.eq_db - self._smoothed_eq_db) * 0.08
        self._smoothed_master_db += (self.master_db - self._smoothed_master_db) * 0.08
        for band_index, (freq, gain_db) in enumerate(zip(EQ_BANDS, self._smoothed_eq_db)):
            if abs(float(gain_db)) >= 0.03:
                out = self._apply_biquad(out, band_index, self._peaking_coefficients(freq, float(gain_db)))
        out *= db_to_amp(self._smoothed_master_db)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.98:
            out *= 0.98 / peak
        return np.clip(out, -1.0, 1.0)

    def _read_deck_chunk(self, deck: DeckState, frames: int) -> np.ndarray:
        if len(deck.audio) == 0:
            chunk = np.zeros((frames, 2), dtype=np.float32)
            self._update_spectrum(deck.spectrum, chunk)
            return chunk

        if deck.scratching:
            start_pos = deck.position
            end_pos = deck.scratch_target
            
            # Zabezpieczenie przed ekstremalnymi dźwiękami (limitujemy x5 prędkość)
            max_speed = 5.0
            max_diff = frames * max_speed
            diff = end_pos - start_pos
            if abs(diff) > max_diff:
                end_pos = start_pos + math.copysign(max_diff, diff)
                
            speed = (end_pos - start_pos) / frames
            
            # Jeśli winyl w miejscu stoi pod palcem (cisza/szum)
            if abs(speed) < 0.01:
                chunk = np.zeros((frames, 2), dtype=np.float32)
                deck.position = end_pos
            else:
                # Interpolacja liniowa ułamków – resampling przyspieszający / zwalniający
                idx = start_pos + np.arange(frames) * speed
                max_idx = len(deck.audio) - 1
                idx_clipped = np.clip(idx, 0, max_idx)
                
                idx_floor = np.floor(idx_clipped).astype(int)
                idx_ceil = np.clip(idx_floor + 1, 0, max_idx)
                frac = (idx_clipped - idx_floor)[:, np.newaxis].astype(np.float32)
                
                # Faktyczne mieszanie dwóch sąsiadujących cyfrowych próbek (powstaje piskliwy dźwięk skreczu)
                chunk = deck.audio[idx_floor] * (1.0 - frac) + deck.audio[idx_ceil] * frac
                deck.position = end_pos
                
            self._update_spectrum(deck.spectrum, chunk)
            return chunk.astype(np.float32)

        if not deck.playing:
            chunk = np.zeros((frames, 2), dtype=np.float32)
            self._update_spectrum(deck.spectrum, chunk)
            return chunk

        # Normalne odtwarzanie bez ruszania winyla
        start_idx = int(deck.position)
        end_idx = min(start_idx + frames, len(deck.audio))
        chunk = deck.audio[start_idx:end_idx].copy()
        deck.position += float(frames)
        
        if end_idx >= len(deck.audio):
            deck.playing = False
            deck.position = float(len(deck.audio) - 1)
            
        if len(chunk) < frames:
            chunk = np.vstack([chunk, np.zeros((frames - len(chunk), 2), dtype=np.float32)])
            
        self._update_spectrum(deck.spectrum, chunk)
        return chunk

    def _callback(self, outdata, frames, _time_info, status):
        del status
        with self.lock:
            deck_a = self._read_deck_chunk(self.decks[0], frames)
            deck_b = self._read_deck_chunk(self.decks[1], frames)
            x = float(self.crossfader)
            gain_a = math.cos(x * math.pi * 0.5) * self.decks[0].volume
            gain_b = math.sin(x * math.pi * 0.5) * self.decks[1].volume
            eq_db = self.eq_db.copy()
            master_db = self.master_db

        self.eq_db = eq_db
        self.master_db = master_db
        mixed = (deck_a * gain_a) + (deck_b * gain_b)
        # --- Miksowanie aktywnych sampli z Launchpada ---
        sampler_mix = np.zeros((frames, 2), dtype=np.float32)
        still_active = []
        for s in self.active_samples:
            pos = s["pos"]
            audio = s["audio"]
            remains = len(audio) - pos
            take = min(frames, remains)
            
            sampler_mix[:take] += audio[pos:pos+take]
            s["pos"] += take
            
            if s["pos"] < len(audio):
                still_active.append(s)
                
        self.active_samples = still_active
        mixed += sampler_mix
        processed = self._master_process(mixed)
        self._update_spectrum(self.spectrum, processed)
        outdata[:] = processed


class SpinningPlatterWidget(QWidget):
    def __init__(self, engine, deck_index):
        super().__init__()
        self.engine = engine
        self.deck_index = deck_index
        self.setMinimumSize(140, 140)
        self.angle = 0.0
        self.cover_pixmap = None
        self.current_track = None
        
        # --- Zmienne do interakcji i scratchowania ---
        self.is_dragging = False
        self.last_mouse_angle = 0.0

    def _get_mouse_angle(self, pos):
        # Oblicza kąt myszki względem środka winyla
        rect = self.rect()
        cx, cy = rect.width() / 2.0, rect.height() / 2.0
        dx = pos.x() - cx
        dy = pos.y() - cy
        return math.degrees(math.atan2(dy, dx))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_angle = self._get_mouse_angle(event.position())
            with self.engine.lock:
                deck = self.engine.decks[self.deck_index]
                deck.scratching = True
                deck.scratch_target = float(deck.position)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            current_angle = self._get_mouse_angle(event.position())
            delta = current_angle - self.last_mouse_angle
            
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
                
            self.angle += delta
            self.last_mouse_angle = current_angle
            
            with self.engine.lock:
                deck = self.engine.decks[self.deck_index]
                if len(deck.audio) > 0:
                    # Mnożnik określa jak długi kawałek odczytujemy (czułość skreczu)
                    sample_shift = delta * 0.015 * 44100
                    deck.scratch_target = np.clip(deck.scratch_target + sample_shift, 0, float(len(deck.audio) - 1))
            
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            with self.engine.lock:
                deck = self.engine.decks[self.deck_index]
                deck.scratching = False
                deck.position = deck.scratch_target
                
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        
        size = min(rect.width(), rect.height())
        cx, cy = rect.width() / 2, rect.height() / 2
        
        status = self.engine.deck_status(self.deck_index)
        
        # Płyta sama się kręci tylko wtedy, gdy muzyka gra i nikt jej NIE trzyma
        if status["playing"] and not self.is_dragging:
            self.angle += 3.0
            if self.angle >= 360:
                self.angle -= 360

        # Wczytanie okładki
        track = status["track"]
        if track != self.current_track:
            self.current_track = track
            self.cover_pixmap = None
            if track and track.cover_path and track.cover_path.exists():
                self.cover_pixmap = QPixmap(str(track.cover_path))
        
        # Tło winyla
        painter.setBrush(QBrush(QColor("#111111")))
        painter.setPen(QPen(QColor("#273244"), 2))
        painter.drawEllipse(int(cx - size/2), int(cy - size/2), int(size), int(size))
        
        # Rowki
        painter.setPen(QPen(QColor("#1a1a1a"), 1))
        for i in range(1, 6):
            r = size/2 - i*10
            if r > 0:
                painter.drawEllipse(int(cx - r), int(cy - r), int(r*2), int(r*2))
        
        # Obrót obszaru roboczego do malowania okładki
        painter.translate(cx, cy)
        painter.rotate(self.angle)
        
        cover_size = int(size * 0.45)
        painter.setBrush(QBrush(QColor("#245070")))
        painter.drawEllipse(int(-cover_size/2), int(-cover_size/2), cover_size, cover_size)
        
        if self.cover_pixmap and not self.cover_pixmap.isNull():
            scaled_cover = self.cover_pixmap.scaled(
                cover_size, cover_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            painter.save()
            
            # Tworzymy okrągłą ścieżkę do wycięcia grafiki
            clip_path = QPainterPath()
            clip_path.addEllipse(int(-cover_size/2), int(-cover_size/2), cover_size, cover_size)
            painter.setClipPath(clip_path)
            
            painter.drawPixmap(int(-cover_size/2), int(-cover_size/2), cover_size, cover_size, scaled_cover)
            painter.restore()
            
        # Otwór na środku winyla
        painter.setBrush(QBrush(QColor("#0c0f14")))
        painter.drawEllipse(-4, -4, 8, 8)

class KnobSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, title: str, minimum: int, maximum: int, value: int, suffix: str = ""):
        super().__init__()
        self.suffix = suffix
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.value_label = QLabel("")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._emit)
        self._emit(value)

    def _emit(self, raw: int):
        value = raw / 10.0
        sign = "+" if value > 0 else ""
        self.value_label.setText(f"{sign}{value:.1f}{self.suffix}")
        self.value_changed.emit(value)

    def set_float_value(self, value: float):
        self.slider.setValue(int(round(value * 10)))


class SpectrumWidget(QWidget):
    def __init__(self, values_getter):
        super().__init__()
        self.values_getter = values_getter
        self.setMinimumHeight(120)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#11151c"))
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor("#49d2ff"))
        gradient.setColorAt(0.55, QColor("#71f79f"))
        gradient.setColorAt(1, QColor("#ffcf5a"))
        values = self.values_getter()
        bar_gap = 2
        bar_width = max(2, (rect.width() - bar_gap * (len(values) - 1)) / len(values))
        for i, value in enumerate(values):
            height = float(value) * (rect.height() - 16)
            x = i * (bar_width + bar_gap)
            y = rect.height() - height
            painter.fillRect(QRectF(x, y, bar_width, height), gradient)
        painter.setPen(QPen(QColor("#273244"), 1))
        for i in range(1, 4):
            y = rect.height() * i / 4
            painter.drawLine(0, y, rect.width(), y)

class MainWindow(QMainWindow):
    download_finished = Signal(str, str)
    bpm_calculated = Signal(QListWidgetItem, str) # NOWY SYGNAŁ

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PulseDeck DJ Master")
        self.resize(1440, 860)
        self.engine = DJAudioEngine()
        self.tracks: list[Track] = []
        self.eq_sliders: list[KnobSlider] = []
        self.transition_active = False
        self.transition_elapsed = 0.0
        self.transition_duration = 1.0
        self.transition_start = 0.0
        self.transition_end = 1.0
        self.transition_preset = TRANSITION_PRESETS["Equal Power Blend"]
        self.transition_base_eq = np.zeros(len(EQ_BANDS), dtype=np.float32)
        self._build_ui()
        self._build_menu()
        self.download_finished.connect(self._download_finished)
        self.bpm_calculated.connect(self._update_bpm_label) # NOWE
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def closeEvent(self, event):
        self.engine.close()
        event.accept()

    def _build_menu(self):
        eq_menu = self.menuBar().addMenu("EQ Preset")
        for name, values in EQ_PRESETS.items():
            action = QAction(name, self)
            action.triggered.connect(lambda _checked=False, v=values: self.apply_eq_preset(v))
            eq_menu.addAction(action)

        transition_menu = self.menuBar().addMenu("DJ Transition")
        for name in TRANSITION_PRESETS:
            action = QAction(name, self)
            action.triggered.connect(lambda _checked=False, n=name: self._select_transition(n))
            transition_menu.addAction(action)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(18, 16, 18, 18)
        main.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("PulseDeck DJ Master")
        title.setObjectName("AppTitle")
        self.status = QLabel("Load tracks into Deck A and Deck B")
        self.status.setObjectName("NowPlaying")
        header.addWidget(title)
        header.addWidget(self.status, 1)
        main.addLayout(header)

        top_splitter = QSplitter(Qt.Horizontal)
        main.addWidget(top_splitter, 1)
        top_splitter.addWidget(self._build_library_panel())
        top_splitter.addWidget(self._build_deck_panel(0, "DECK A"))
        top_splitter.addWidget(self._build_deck_panel(1, "DECK B"))
        top_splitter.setSizes([320, 560, 560])

        main.addWidget(self._build_mixer_panel())
        self.setStyleSheet(STYLE)

    def _build_library_panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        title = QLabel("Library / Queue")
        title.setObjectName("PanelTitle")
        
        # --- WYSZUKIWARKA ---
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Wyszukaj piosenkę...")
        self.search_input.textChanged.connect(self._filter_library)
        
        self.queue = QListWidget()
        self.queue.itemDoubleClicked.connect(lambda item: self.load_selected_to_deck(0, autoplay=True))

        add_button = QPushButton("Add Audio Files")
        add_button.clicked.connect(self.add_tracks)
        load_a = QPushButton("Load Selected -> A")
        load_a.clicked.connect(lambda: self.load_selected_to_deck(0))
        load_b = QPushButton("Load Selected -> B")
        load_b.clicked.connect(lambda: self.load_selected_to_deck(1))
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_selected)

        self.youtube_input = QLineEdit()
        self.youtube_input.setPlaceholderText("YouTube link")
        download = QPushButton("Download Audio")
        download.clicked.connect(self.download_youtube)

        layout.addWidget(title)
        layout.addWidget(self.search_input) # DODANO
        layout.addWidget(self.queue, 1)
        layout.addWidget(add_button)
        row = QHBoxLayout()
        row.addWidget(load_a)
        row.addWidget(load_b)
        layout.addLayout(row)
        layout.addWidget(remove)
        layout.addWidget(QLabel("YouTube import"))
        layout.addWidget(self.youtube_input)
        layout.addWidget(download)
        suggest_layout = QHBoxLayout()
        match_a_btn = QPushButton("Match to Deck A")
        match_b_btn = QPushButton("Match to Deck B")
        match_a_btn.clicked.connect(lambda: self.suggest_matching(0))
        match_b_btn.clicked.connect(lambda: self.suggest_matching(1))
        
        suggest_layout.addWidget(match_a_btn)
        suggest_layout.addWidget(match_b_btn)
        layout.addWidget(QLabel("BPM Match"))
        layout.addLayout(suggest_layout)
        
        return panel

    def _build_deck_panel(self, deck_index: int, title_text: str):
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        title = QLabel(title_text)
        title.setObjectName("PanelTitle")
        track_label = QLabel("No track")
        track_label.setObjectName("DeckTitle")
        spectrum = SpectrumWidget(lambda i=deck_index: self.engine.deck_status(i)["spectrum"])
        platter = SpinningPlatterWidget(self.engine, deck_index) 
        
        # Układ ułożenia poziomego widma i obracającej się płyty
        visual_layout = QHBoxLayout()
        visual_layout.addWidget(platter)
        visual_layout.addWidget(spectrum, 1)

        play = self._tool_button(self._standard_icon("SP_MediaPlay"), lambda checked=False, i=deck_index: self.toggle_deck(i))
        cue = QPushButton("Cue")
        cue.clicked.connect(lambda _checked=False, i=deck_index: self.cue_deck(i))
        progress = QSlider(Qt.Horizontal)
        progress.setRange(0, 10000)
        progress.sliderMoved.connect(lambda value, i=deck_index: self.engine.seek_deck_ratio(i, value / 10000.0))
        time_label = QLabel("0:00 / 0:00")
        volume_label, volume_box = self._horizontal_control(
            "Deck Volume", 0, 15, 10, "x", lambda value, i=deck_index: self.engine.set_deck_volume(i, value)
        )

        if deck_index == 0:
            self.deck_a_label = track_label
            self.deck_a_play = play
            self.deck_a_progress = progress
            self.deck_a_time = time_label
            self.deck_a_spectrum = spectrum
            self.deck_a_platter = platter # NOWE
        else:
            self.deck_b_label = track_label
            self.deck_b_play = play
            self.deck_b_progress = progress
            self.deck_b_time = time_label
            self.deck_b_spectrum = spectrum
            self.deck_b_platter = platter # NOWE

        transport = QHBoxLayout()
        transport.addStretch()
        transport.addWidget(play)
        transport.addWidget(cue)
        transport.addStretch()
        layout.addWidget(title)
        layout.addWidget(track_label)
        layout.addWidget(title)
        layout.addWidget(track_label)
        layout.addLayout(visual_layout) # ZAMIAST layout.addWidget(spectrum)
        layout.addLayout(transport)
        layout.addWidget(progress)
        layout.addWidget(time_label)
        layout.addWidget(volume_label)
        layout.addWidget(volume_box)
        return panel

    def _build_mixer_panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QGridLayout(panel)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        mixer_title = QLabel("Mixer / Transitions")
        mixer_title.setObjectName("PanelTitle")
        self.master_spectrum = SpectrumWidget(lambda: self.engine.spectrum)
        self.crossfader = QSlider(Qt.Horizontal)
        self.crossfader.setRange(0, 1000)
        self.crossfader.setValue(0)
        self.crossfader.valueChanged.connect(lambda value: self.engine.set_crossfader(value / 1000.0))
        self.crossfader_label = QLabel("Crossfader: A")

        self.transition_combo = QComboBox()
        self.transition_combo.addItems(TRANSITION_PRESETS.keys())
        self.transition_combo.currentTextChanged.connect(self._select_transition)
        self.transition_seconds = QSpinBox()
        self.transition_seconds.setRange(1, 90)
        self.transition_seconds.setValue(16)
        start_ab = QPushButton("Transition A -> B")
        start_ab.clicked.connect(lambda: self.start_transition(0, 1))
        start_ba = QPushButton("Transition B -> A")
        start_ba.clicked.connect(lambda: self.start_transition(1, 0))

        self.master = self._horizontal_control("Master", -120, 60, 0, " dB", self._set_master)

        eq_box = QFrame()
        eq_layout = QVBoxLayout(eq_box)
        eq_layout.setContentsMargins(0, 0, 0, 0)
        eq_title = QLabel("Master EQ")
        eq_title.setObjectName("PanelTitle")
        sliders = QHBoxLayout()
        for index, band in enumerate(EQ_BANDS):
            label = f"{band // 1000}k" if band >= 1000 else str(band)
            slider = KnobSlider(label, -120, 120, 0, " dB")
            slider.value_changed.connect(lambda value, i=index: self._set_eq(i, value))
            self.eq_sliders.append(slider)
            sliders.addWidget(slider)
        preset_row = QHBoxLayout()
        self.eq_combo = QComboBox()
        self.eq_combo.addItems(EQ_PRESETS.keys())
        self.eq_combo.currentTextChanged.connect(
            lambda text: self.apply_eq_preset(EQ_PRESETS[text]) if text in EQ_PRESETS else None
        )
        save = QPushButton("Save EQ")
        save.clicked.connect(self.save_eq_preset)
        load = QPushButton("Load EQ")
        load.clicked.connect(self.load_eq_preset)
        reset = QPushButton("Reset EQ")
        reset.clicked.connect(lambda: self.apply_eq_preset(EQ_PRESETS["Flat"]))
        preset_row.addWidget(save)
        preset_row.addWidget(load)
        preset_row.addWidget(reset)
        eq_layout.addWidget(eq_title)
        eq_layout.addLayout(sliders, 1)
        eq_layout.addLayout(preset_row)

        layout.addWidget(mixer_title, 0, 0, 1, 3)
        layout.addWidget(self.master_spectrum, 1, 0, 1, 3)
        layout.addWidget(QLabel("A"), 2, 0)
        layout.addWidget(self.crossfader, 2, 1)
        layout.addWidget(QLabel("B"), 2, 2)
        layout.addWidget(self.crossfader_label, 3, 1)
        layout.addWidget(QLabel("Preset"), 4, 0)
        layout.addWidget(self.transition_combo, 4, 1)
        layout.addWidget(self.transition_seconds, 4, 2)
        layout.addWidget(start_ab, 5, 0)
        layout.addWidget(start_ba, 5, 1)
        layout.addWidget(self.master[0], 6, 0)
        layout.addWidget(self.master[1], 6, 1, 1, 2)
        layout.addWidget(eq_box, 0, 3, 7, 1)
        lp_box = QFrame()
        lp_layout = QGridLayout(lp_box)
        lp_layout.setContentsMargins(0, 0, 0, 0)
        lp_title = QLabel("Launchpad (Klawisze 1-8)")
        lp_title.setObjectName("PanelTitle")
        lp_layout.addWidget(lp_title, 0, 0, 1, 4)
        
        self.pad_buttons = []
        keys = ['1', '2', '3', '4', '5', '6', '7', '8']
        default_names = ["Kick", "Hi-Hat", "Laser", "Pusty", "Pusty", "Pusty", "Pusty", "Pusty"]
        
        for i in range(8):
            btn = QPushButton(f"{i+1}. {default_names[i]}")
            btn.setMinimumHeight(45)
            # Stylowanie przypominające sprzętowy kontroler
            btn.setStyleSheet("background-color: #2b3a4f; border-radius: 8px; font-weight: bold;")
            
            # Lewy przycisk -> gra dźwięk
            btn.clicked.connect(lambda checked=False, idx=i: self._flash_and_play_pad(idx))
            
            # Prawy przycisk -> menu ładowania pliku z dysku (np. śmiesznego tekstu lub nowej stopy)
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, idx=i: self._load_pad_sample(idx))
            
            # Skrót klawiszowy z głównej klawiatury komputera
            shortcut = QShortcut(QKeySequence(keys[i]), self)
            shortcut.activated.connect(lambda idx=i: self._flash_and_play_pad(idx))
            
            self.pad_buttons.append(btn)
            lp_layout.addWidget(btn, 1 + (i//4), i%4)
            
        layout.addWidget(lp_box, 0, 4, 7, 2)
        
        return panel

    def _standard_icon(self, name: str):
        if hasattr(QStyle, name):
            return getattr(QStyle, name)
        return getattr(QStyle.StandardPixmap, name)

    def _tool_button(self, icon_name, slot):
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon_name))
        size = button.iconSize()
        button.setIconSize(QSize(int(size.width() * 1.45), int(size.height() * 1.45)))
        button.clicked.connect(slot)
        return button

    def _horizontal_control(self, name, minimum, maximum, value, suffix, slot):
        label = QLabel(name)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QLabel("")
        value_label.setMinimumWidth(70)

        def update(raw):
            val = raw / 10.0
            value_label.setText(f"{val:.1f}{suffix}")
            slot(val)

        slider.valueChanged.connect(update)
        update(value)
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return label, box

    def add_tracks(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose audio files",
            "",
            "Audio files (*.mp3 *.wav *.flac *.ogg *.m4a *.webm *.opus)",
        )
        for raw_path in paths:
            self._add_track(Path(raw_path))
        if self.tracks:
            self.status.setText(f"{len(self.tracks)} tracks in library")

    def _add_track(self, path: Path):
        track = self._read_track(path)
        self.tracks.append(track)
        item = QListWidgetItem(track.label)
        item.setData(Qt.UserRole, len(self.tracks) - 1)
        self.queue.addItem(item)
        
        # Uruchom analizę BPM w tle, by nie zamrozić programu
        threading.Thread(target=self._calc_bpm_worker, args=(track, item), daemon=True).start()

    def _filter_library(self, text: str):
        search_term = text.lower()
        for i in range(self.queue.count()):
            item = self.queue.item(i)
            item.setHidden(search_term not in item.text().lower())

    def _calc_bpm_worker(self, track: Track, item: QListWidgetItem):
        try:
            import librosa
            # Pobieramy próbki (to może potrwać parę sekund)
            samples = decode_audio(track.path)
            # Librosa preferuje dźwięk mono do detekcji bitów
            mono = np.mean(samples, axis=1)
            
            # Właściwa analiza BPM
            tempo, _ = librosa.beat.beat_track(y=mono, sr=SAMPLE_RATE)
            
            if isinstance(tempo, np.ndarray):
                bpm = float(tempo[0])
            else:
                bpm = float(tempo)
                
            track.bpm = round(bpm)
            # Wysłanie sygnału do zaktualizowania wpisu w głównym wątku UI
            self.bpm_calculated.emit(item, track.label)
        except Exception as e:
            print(f"Błąd podczas obliczania BPM: {e}")

    def _update_bpm_label(self, item: QListWidgetItem, new_label: str):
        item.setText(new_label)

    def suggest_matching(self, deck_index: int):
        status = self.engine.deck_status(deck_index)
        track = status["track"]
        
        if not track or not track.bpm:
            QMessageBox.information(self, "Brak danych", f"Deck {'A' if deck_index == 0 else 'B'} jest pusty lub BPM wciąż się oblicza.")
            return

        target_bpm = track.bpm
        
        # Sortowanie biblioteki wg różnicy BPM. Utwory bez obliczonego BPM lądują na końcu.
        self.tracks.sort(key=lambda t: abs(t.bpm - target_bpm) if t.bpm else 9999)

        # Odświeżenie listy z podświetleniem świetnych dopasowań (+/- 4 BPM)
        self.queue.clear()
        for i, t in enumerate(self.tracks):
            item = QListWidgetItem(t.label)
            if t.bpm and abs(t.bpm - target_bpm) <= 4:
                item.setBackground(QColor("#245028")) # Zielonkawe tło dla idealnych kandydatek
            item.setData(Qt.UserRole, i)
            self.queue.addItem(item)

    def remove_selected(self):
        row = self.queue.currentRow()
        if row >= 0:
            self.queue.takeItem(row)
            del self.tracks[row]
            for i in range(self.queue.count()):
                self.queue.item(i).setData(Qt.UserRole, i)

    def _read_track(self, path: Path) -> Track:
        meta = MutagenFile(path, easy=True)
        title = path.stem
        artist = ""
        duration = 0.0
        if meta is not None:
            title = (meta.get("title") or [title])[0]
            artist = (meta.get("artist") or [""])[0]
            if meta.info is not None:
                duration = float(getattr(meta.info, "length", 0.0) or 0.0)
                
        # Szukanie okładki/miniaturki (yt-dlp zapisuje np. jako .webp lub .jpg)
        cover_path = None
        for ext in ['.webp', '.jpg', '.png', '.jpeg']:
            possible_cover = path.with_suffix(ext)
            if possible_cover.exists():
                cover_path = possible_cover
                break

        return Track(path=path, title=title, artist=artist, duration=duration, cover_path=cover_path)

    def load_selected_to_deck(self, deck_index: int, autoplay: bool = False):
        row = self.queue.currentRow()
        if row < 0:
            QMessageBox.information(self, "No track selected", "Select a track in the library first.")
            return
        track = self.tracks[int(self.queue.item(row).data(Qt.UserRole))]
        try:
            self.engine.load_deck(deck_index, track, autoplay=autoplay)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot load file", str(exc))
            return
        label = self.deck_a_label if deck_index == 0 else self.deck_b_label
        label.setText(track.label)
        self.status.setText(f"Loaded {track.title} to Deck {'A' if deck_index == 0 else 'B'}")

    def toggle_deck(self, deck_index: int):
        playing = self.engine.toggle_deck(deck_index)
        button = self.deck_a_play if deck_index == 0 else self.deck_b_play
        icon = self._standard_icon("SP_MediaPause" if playing else "SP_MediaPlay")
        button.setIcon(self.style().standardIcon(icon))

    def cue_deck(self, deck_index: int):
        self.engine.seek_deck_ratio(deck_index, 0.0)
        self.engine.set_deck_playing(deck_index, False)
        button = self.deck_a_play if deck_index == 0 else self.deck_b_play
        button.setIcon(self.style().standardIcon(self._standard_icon("SP_MediaPlay")))

    def _select_transition(self, name: str):
        preset = TRANSITION_PRESETS[name]
        self.transition_preset = preset
        self.transition_seconds.setValue(int(preset["duration"]))
        self.status.setText(preset["description"])

    def start_transition(self, from_deck: int, to_deck: int):
        status = self.engine.deck_status(to_deck)
        if status["track"] is None:
            QMessageBox.information(self, "Target deck is empty", "Load a track into the target deck first.")
            return
        self.engine.set_deck_playing(to_deck, True)
        self.transition_active = True
        self.transition_elapsed = 0.0
        self.transition_duration = float(self.transition_seconds.value())
        self.transition_start = 0.0 if from_deck == 0 else 1.0
        self.transition_end = 1.0 if to_deck == 1 else 0.0
        self.transition_base_eq = self.engine.eq_db.copy()
        self.status.setText(f"Transition {'A -> B' if to_deck == 1 else 'B -> A'}: {self.transition_combo.currentText()}")

    def _transition_curve(self, t: float) -> float:
        curve = self.transition_preset.get("curve", "equal_power")
        if curve == "sharp":
            return 0.0 if t < 0.48 else 1.0 if t > 0.52 else (t - 0.48) / 0.04
        if curve == "s_curve":
            return t * t * (3.0 - 2.0 * t)
        return t

    def _apply_transition_eq(self, t: float):
        shaped = math.sin(math.pi * t)
        eq = self.transition_base_eq.copy()
        if "bass_dip" in self.transition_preset:
            eq[0:3] += float(self.transition_preset["bass_dip"]) * shaped
        if "high_dip" in self.transition_preset:
            eq[7:10] += float(self.transition_preset["high_dip"]) * shaped
        with self.engine.lock:
            self.engine.eq_db = eq

    def download_youtube(self):
        url = self.youtube_input.text().strip()
        if not url:
            return
        self.status.setText("Downloading YouTube audio...")
        thread = threading.Thread(target=self._download_youtube_worker, args=(url,), daemon=True)
        thread.start()

    def _download_youtube_worker(self, url: str):
        try:
            import yt_dlp

            output_dir = Path.cwd() / "downloads"
            output_dir.mkdir(exist_ok=True)
            options = {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": str(output_dir / "%(title).180s.%(ext)s"),
                "quiet": True,
                "noplaylist": True,
                "writethumbnail": True, # <-- DODANO: pobieranie miniaturki z YT
            }
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
                prepared = Path(downloader.prepare_filename(info))
            self.download_finished.emit(str(prepared), "")
        except Exception as exc:
            self.download_finished.emit("", str(exc))

    def _download_finished(self, path: str, error: str):
        if error:
            QMessageBox.critical(self, "YouTube download failed", error)
            self.status.setText("YouTube download failed")
            return
        track_path = Path(path)
        self._add_track(track_path)
        self.status.setText(f"Downloaded: {track_path.name}")

    def _set_eq(self, index: int, value: float):
        with self.engine.lock:
            self.engine.eq_db[index] = value

    def _set_master(self, value: float):
        with self.engine.lock:
            self.engine.master_db = value

    def apply_eq_preset(self, values):
        for slider, value in zip(self.eq_sliders, values):
            slider.set_float_value(value)

    def save_eq_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save EQ preset", "preset.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "bands": EQ_BANDS,
            "values_db": [float(slider.slider.value() / 10.0) for slider in self.eq_sliders],
            "master_db": self.engine.master_db,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_eq_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load EQ preset", "", "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.apply_eq_preset(data.get("values_db", EQ_PRESETS["Flat"]))

    def _flash_and_play_pad(self, index: int):
        self.engine.play_sample(index)
        btn = self.pad_buttons[index]
        original_style = btn.styleSheet()
        btn.setStyleSheet("background-color: #ffcf5a; color: black; border-radius: 8px; font-weight: bold;")
        # Szybki reset koloru padu by zasymulować błyśnięcie
        QTimer.singleShot(100, lambda: btn.setStyleSheet(original_style))

    def _load_pad_sample(self, index: int):
        path, _ = QFileDialog.getOpenFileName(self, f"Załaduj dźwięk na Pad {index+1}", "", "Audio (*.mp3 *.wav *.flac)")
        if path:
            success = self.engine.load_sample(index, Path(path))
            if success:
                self.pad_buttons[index].setText(Path(path).stem[:8] + "..")
                self.pad_buttons[index].setStyleSheet("background-color: #49d2ff; color: black; border-radius: 8px; font-weight: bold;")

    def _tick(self):
        for index in range(2):
            status = self.engine.deck_status(index)
            progress = self.deck_a_progress if index == 0 else self.deck_b_progress
            time_label = self.deck_a_time if index == 0 else self.deck_b_time
            play_button = self.deck_a_play if index == 0 else self.deck_b_play
            if status["duration"] > 0 and not progress.isSliderDown():
                progress.setValue(int((status["current"] / status["duration"]) * 10000))
            time_label.setText(f"{self._fmt_time(status['current'])} / {self._fmt_time(status['duration'])}")
            icon_name = "SP_MediaPause" if status["playing"] else "SP_MediaPlay"
            play_button.setIcon(self.style().standardIcon(self._standard_icon(icon_name)))

        if self.transition_active:
            self.transition_elapsed += self.timer.interval() / 1000.0
            t = min(1.0, self.transition_elapsed / self.transition_duration)
            curved = self._transition_curve(t)
            value = self.transition_start + (self.transition_end - self.transition_start) * curved
            self.crossfader.blockSignals(True)
            self.crossfader.setValue(int(value * 1000))
            self.crossfader.blockSignals(False)
            self.engine.set_crossfader(value)
            self._apply_transition_eq(t)
            if t >= 1.0:
                self.transition_active = False
                with self.engine.lock:
                    self.engine.eq_db = self.transition_base_eq.copy()

        cf = self.engine.crossfader
        self.crossfader_label.setText(f"Crossfader: {int((1.0 - cf) * 100)}% A / {int(cf * 100)}% B")
        self.deck_a_spectrum.update()
        self.deck_b_spectrum.update()
        self.deck_a_platter.update() # NOWE Odświeżanie płyty A
        self.deck_b_platter.update() # NOWE Odświeżanie płyty B
        self.master_spectrum.update()

    @staticmethod
    def _fmt_time(value: float) -> str:
        minutes = int(value // 60)
        seconds = int(value % 60)
        return f"{minutes}:{seconds:02d}"


STYLE = """
QMainWindow, QWidget {
    background: #0c0f14;
    color: #e6edf3;
    font-family: Segoe UI, Arial;
    font-size: 13px;
}
QMenuBar, QMenu {
    background: #141922;
    color: #e6edf3;
}
QMenu::item:selected {
    background: #263244;
}
#AppTitle {
    font-size: 30px;
    font-weight: 700;
    color: #f2f7ff;
}
#NowPlaying {
    color: #9fb0c7;
    font-size: 15px;
}
#Panel {
    background: #141922;
    border: 1px solid #263244;
    border-radius: 8px;
}
#PanelTitle {
    font-size: 16px;
    font-weight: 700;
}
#DeckTitle {
    color: #a8bdd8;
    font-size: 14px;
}
QLineEdit, QComboBox, QSpinBox {
    background: #0f131a;
    border: 1px solid #263244;
    border-radius: 6px;
    padding: 7px;
    color: #e6edf3;
}
QListWidget {
    background: #0f131a;
    border: 1px solid #263244;
    border-radius: 6px;
    padding: 6px;
}
QListWidget::item {
    padding: 9px;
    border-radius: 5px;
}
QListWidget::item:selected {
    background: #245070;
}
QPushButton, QToolButton {
    background: #1f6feb;
    color: white;
    border: 0;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 600;
}
QPushButton:hover, QToolButton:hover {
    background: #2b7fff;
}
QToolButton {
    min-width: 46px;
    min-height: 42px;
}
QSlider::groove:horizontal {
    height: 7px;
    background: #273244;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
    background: #71f79f;
}
QSlider::groove:vertical {
    width: 7px;
    background: #273244;
    border-radius: 3px;
}
QSlider::handle:vertical {
    height: 18px;
    margin: 0 -6px;
    border-radius: 9px;
    background: #49d2ff;
}
QSplitter::handle {
    background: #0c0f14;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PulseDeck DJ Master")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
