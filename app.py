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
from PySide6.QtGui import QAction, QColor, QLinearGradient, QPainter, QPen
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
    "Club": [5, 4, 2, 0, 0, 2, 4, 5, 3, 2],
    "Vocal": [-2, -1, 0, 2, 4, 5, 4, 2, 0, -1],
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

    @property
    def label(self) -> str:
        left = f"{self.artist} - {self.title}" if self.artist else self.title
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{left}   {minutes}:{seconds:02d}"


class DeckState:
    def __init__(self, name: str):
        self.name = name
        self.audio = np.zeros((0, 2), dtype=np.float32)
        self.position = 0
        self.playing = False
        self.track: Track | None = None
        self.volume = 1.0
        self.spectrum = np.zeros(128, dtype=np.float32)

    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    def current_time(self) -> float:
        return self.position / SAMPLE_RATE


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

    def close(self):
        self.stream.stop()
        self.stream.close()

    def load_deck(self, deck_index: int, track: Track, autoplay: bool = False):
        samples = decode_audio(track.path)
        with self.lock:
            deck = self.decks[deck_index]
            deck.audio = samples
            deck.position = 0
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
                deck.position = int(np.clip(ratio, 0.0, 1.0) * (len(deck.audio) - 1))

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
        if not deck.playing or len(deck.audio) == 0:
            chunk = np.zeros((frames, 2), dtype=np.float32)
            self._update_spectrum(deck.spectrum, chunk)
            return chunk

        end = min(deck.position + frames, len(deck.audio))
        chunk = deck.audio[deck.position:end].copy()
        deck.position = end
        if end >= len(deck.audio):
            deck.playing = False
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
        processed = self._master_process(mixed)
        self._update_spectrum(self.spectrum, processed)
        outdata[:] = processed


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


class MainWindow(QMainWindow):
    download_finished = Signal(str, str)

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
        play = self._tool_button(self._standard_icon("SP_MediaPlay"), lambda i=deck_index: self.toggle_deck(i))
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
        else:
            self.deck_b_label = track_label
            self.deck_b_play = play
            self.deck_b_progress = progress
            self.deck_b_time = time_label
            self.deck_b_spectrum = spectrum

        transport = QHBoxLayout()
        transport.addStretch()
        transport.addWidget(play)
        transport.addWidget(cue)
        transport.addStretch()
        layout.addWidget(title)
        layout.addWidget(track_label)
        layout.addWidget(spectrum)
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
        return Track(path=path, title=title, artist=artist, duration=duration)

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
