# AntiJomok 🖐️✨

Portal filter real-time pakai gerakan tangan, dibikin pake Python + MediaPipe.

Bentangin dua tangan buat buka portal, area di dalamnya kena filter (dual-tone, thermal, sketch, glitch, dll). Ganti filter tinggal pinch jempol-kelingking.

---

## 🇮🇩 Indonesia

**Install** (Python 3.8–3.11):
```bash
pip install -r requirements.txt
```
> ⚠️ Pengguna Apple Silicon: jangan upgrade mediapipe dari versi yang di-pin (`0.10.9`) — versi baru ada bug di chip ARM Mac.

**Jalanin:**
```bash
python3 jomok.py   # Mac/Linux
python jomok.py    # Windows
```

**Kontrol:**
- Bentangin 2 tangan → buka portal
- Pinch jempol-kelingking → ganti filter
- Kepal 2 tangan / tombol `C` → toggle mode 2D/3D
- `N`/`P` → filter berikutnya/sebelumnya, `S` → screenshot, `Q` → keluar


**Mode Meme** (default, tombol `M` buat pindah ke mode Portal) — tahan gesture sebentar, memenya muncul di atas tangan:
| Gesture | Meme |
|---|---|
| 👍 jempol | kucing jempol |
| 🖕 jari tengah | kucing jari tengah |
| ✌️ peace | anak lelaki |
| ☝️ nunjuk | izin berpikir |
| 👌 OK | #nggakapaapa |
| 🤘 metal | Son 😭 |
| 2 tangan terbuka | sopo iki cik? |
| 2 tangan dikepal | aku ngerti iki sopo!! |
| 2 tangan nutup muka | takut |

Gambar meme ada di folder `memes/`, ganti aja filenya kalau mau meme lain.

Dibikin sambil belajar OpenCV + MediaPipe.

---

## 🇬🇧 English

**Install** (Python 3.8–3.11):
```bash
pip install -r requirements.txt
```
> ⚠️ Apple Silicon users: don't upgrade mediapipe past the pinned version (`0.10.9`) — newer releases are buggy on ARM Macs.

**Run:**
```bash
python3 jomok.py   # Mac/Linux
python jomok.py    # Windows
```

**Controls:**
- Spread both hands → open portal
- Pinch thumb + pinky → switch filter
- Fist both hands / press `C` → toggle 2D/3D mode
- `N`/`P` → next/previous filter, `S` → screenshot, `Q` → quit


**Meme mode** (default, press `M` to switch to Portal mode) — hold a gesture briefly and a meme pops up above your hand: 👍 👌 ✌️ ☝️ 🤘 🖕, both hands open, both fists, or both hands covering your face. Meme images live in `memes/`.

Built while learning OpenCV + MediaPipe.

---

## Hall of Jomok

- FARIS JOMOK

- PARSHA JOMOK

- NADHIF NOJOMOK

---

MIT License
