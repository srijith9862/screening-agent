# Screening Agent

An AI voice agent that makes first-round screening calls **through your own Android phone and SIM** (built for a Galaxy S24 Ultra + Windows laptop). It follows a question script, answers candidate questions from a fixed FAQ, records the call (with consent), and writes a scorecard for every candidate.

```
Windows laptop (this program)                       Galaxy S24 Ultra
  dials via ADB (USB)  ───────────────────────────▶  places the call on your SIM
  candidate audio  ◀── Phone Link (Bluetooth) ──────  call audio
  agent voice      ───────────────────────────────▶  to the candidate
  Deepgram (speech ↔ text)  +  Claude (conversation, scorecard)
  output/: recording.wav, transcript.json, scorecard.md, results.csv
```

- One call at a time. A typical screen is ~12 minutes.
- The agent says up front that it is an AI and the call is recorded, and asks for consent. If the candidate says no, the recording and transcript are **deleted** and they are marked for a personal call.
- The agent never decides. It recommends advance / hold / reject; you decide.

---

## 1. One-time setup on Windows

### Software
1. **Python 3.11+** from python.org (tick "Add python.exe to PATH").
2. **Android platform-tools** (ADB): download from developer.android.com/tools/releases/platform-tools, unzip to `C:\platform-tools`. Either add it to PATH or set `phone.adb_path: C:\platform-tools\adb.exe` in `config/settings.yaml`.
3. **Two virtual audio cables** from vb-audio.com: **VB-CABLE** (free) and **VB-CABLE A+B** (donationware). Reboot after installing.
4. **Phone Link** (preinstalled on Windows 11) paired with the phone, with **Calls** set up (it pairs over Bluetooth).

### Install
```bash
git clone https://github.com/srijith9862/screening-agent
```
```bash
cd screening-agent
```
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

### API keys
Copy `.env.example` to `.env` and fill in:
- `ANTHROPIC_API_KEY` from console.anthropic.com
- `DEEPGRAM_API_KEY` from console.deepgram.com (speech-to-text and voice)

### Phone (S24 Ultra)
1. Settings → About phone → Software information → tap **Build number** 7 times to unlock Developer options.
2. Settings → Developer options → turn on **USB debugging** and **Stay awake**.
3. Connect by USB, tap **Allow** on the "Allow USB debugging?" prompt (tick "Always allow").
4. Keep the phone unlocked and on charge while calling.

### Route the call audio
Windows Settings → System → Sound → **Volume mixer** → **Phone Link**:
- **Output device:** `CABLE Input (VB-Audio Virtual Cable)`, so the candidate's voice goes to the agent
- **Input device:** `CABLE-A Output (VB-Audio Cable A)`, so the agent's voice goes into the call

Want to listen in live? Control Panel → Sound → Recording → **CABLE Output** → Properties → Listen → tick **Listen to this device** and pick your headphones.

If your device names differ, run `python -m screener devices` and update the `audio:` section of `config/settings.yaml`.

### Check everything
```bash
python -m screener check
```
Then call a friend from Phone Link and run:
```bash
python -m screener audio-test
```
Your friend should hear a beep, and the level bar should move when they talk.

> **If Phone Link ignores the per-app setting**, set the same two devices as the Windows *default communication devices* (Control Panel → Sound → right-click → Set as Default Communication Device) and try again.

---

## 2. Before calling: fill in your details

- `config/faq.yaml`: replace every `TODO` (office hours, interview process, benefits...). Until then, the agent tells candidates "the HR team will get back to you" for those topics.
- `config/settings.yaml` → `script_values`: office schedule and US-overlap hours.
- `config/script.yaml`: the questions. Edit freely.
- Put your Naukri shortlist at `data/candidates.csv`. It needs a **`Phone`** column (`+919876543210` or a 10-digit number). An optional `Preferred_call_time` column is shown before each call. `Category` and `Rating_out_of_10` are deliberately **not** shown to the AI, so it isn't biased.

`data/` (except the sample) and `output/` are git-ignored: candidate data never goes to GitHub.

---

## 3. Practise first

**Text practice** (only needs the Anthropic key). You type as the candidate:
```bash
python -m screener practice --name "Harshitha"
```

**Voice rehearsal** on your laptop mic. **Use headphones**, otherwise the agent hears itself:
```bash
python -m screener rehearse --name "Harshitha"
```

Both write a scorecard to `output/` so you can check the scoring too.

---

## 4. Make calls

```bash
python -m screener call
```
- Asks before each call (`Y` / `s`kip / `q`uit). Add `--yes` to stop asking, `--name "Kiran"` to call one person.
- Only calls inside `calling_hours` (10:00–19:00 IST by default) and waits `gap_between_calls_s` between calls.
- Skips anyone already screened (outcome completed / declined / not interested). `--redo` calls them again. No-answers and call-backs are retried on the next run.
- Press `Ctrl+C` to stop at any time; the phone call is hung up.

### Outputs
- `output/results.csv`: one row per call (open in Excel): outcome, recommendation, section scores, knockouts, call-back requests, summary.
- `output/calls/<date>/<name>-<time>/`
  - `scorecard.md`: the readable scorecard
  - `scorecard.json`, `transcript.json`, `call.json`
  - `recording.wav`: stereo, left = candidate, right = agent (only if they consented)

Re-run a scorecard after editing the rubric:
```bash
python -m screener score output\calls\2026-11-07\harshitha-devulapalli-103015
```

### Retention
Recordings are personal data (DPDP Act 2023). Delete call folders older than `retention_days` (90 by default):
```bash
python -m screener purge
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Agent starts talking during the caller tune / never starts | Run `python -m screener adb-state` during a call. If you don't see `state: ACTIVE` lines once answered, set `phone.answer_detection: speech`. |
| Candidate can't hear the agent | Phone Link input device must be `CABLE-A Output`; `audio.call_out_device` must be `CABLE-A Input`. Re-run `audio-test`. |
| Agent can't hear the candidate | Phone Link output device must be `CABLE Input`; `audio.call_in_device` must be `CABLE Output`. |
| Replies feel slow | Lower `deepgram.endpointing_ms` (e.g. 300). `claude.live_effort` is already `low`. |
| Agent interrupts the candidate mid-thought | Raise `deepgram.utterance_end_ms` (e.g. 1500) and `endpointing_ms` (e.g. 600). |
| `en-IN` errors from Deepgram | Set `deepgram.stt_language: en`. |
| Number gets flagged as spam | Increase `gap_between_calls_s`, and message candidates on Naukri/WhatsApp before calling. |

## Tests
```bash
pip install -r requirements-dev.txt
```
```bash
python -m pytest
```
