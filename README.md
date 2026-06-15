# PulseDeck DJ Master

Zaawansowana konsola DJ napisana w Pythonie:

- dwa decki A/B z niezaleznym start/stop, cue, paskiem postepu i glosnoscia
- kolejka/library i ladowanie wybranego utworu do Deck A albo Deck B
- crossfader equal-power do miksowania dwoch utworow
- automatyczne przejscia DJ z presetow
- pobieranie audio z linku YouTube przez `yt-dlp` do folderu `downloads`
- 10-pasmowy master equalizer dzialajacy w czasie rzeczywistym
- plynna zmiana pasm bez trzaskow przez filtry biquad peaking EQ
- master gain i limiter
- wizualizery widma FFT dla Deck A, Deck B i master output
- zapisywanie i wczytywanie presetow EQ
- ciemny interfejs PySide6
- dekodowanie MP3 przez dolaczony FFmpeg z paczki `imageio-ffmpeg`

## Instalacja

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Do dekodowania audio aplikacja uzywa `imageio-ffmpeg`, wiec nie trzeba osobno instalowac FFmpeg w systemie. Import z YouTube zapisuje najlepszy dostepny plik audio bez konwersji, zwykle `.m4a` albo `.webm`.
Pobieranie z YouTube uzywa `yt-dlp`; korzystaj tylko z materialow, ktore masz prawo pobrac.

Po instalacji otworz nowy terminal i uruchom:

```powershell
py app.py
```

## Sterowanie

- `Add Audio Files` dodaje utwory do biblioteki.
- `Load Selected -> A/B` laduje wybrany utwor do decka.
- `Transition A -> B` i `Transition B -> A` uruchamiaja automatyczne przejscie.
- `YouTube link` + `Download Audio` pobiera audio do `downloads` i dodaje je do biblioteki.
- Suwaki EQ zmieniaja master EQ w locie.

## Presety przejsc

- `Equal Power Blend` - klasyczne gladkie przejscie crossfaderem.
- `Smooth S-Curve` - dluzsze klubowe przejscie po krzywej S.
- `Bass Swap` - przejscie z chwilowym obnizeniem basu, zeby kicki sie nie nakladaly.
- `Filter Fade` - przejscie z delikatnym przygaszeniem wysokich pasm.
- `Quick Cut` - szybkie ciecie pod drop albo mocny punkt frazy.
