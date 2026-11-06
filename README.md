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

### Step 1: Get the code
Open **PowerShell** (Start menu → type `PowerShell` → Enter) and run these one at a time:
```bash
winget install -e --id Git.Git
```
Close and reopen PowerShell, then:
```bash
cd $HOME\Documents
```
```bash
git clone https://github.com/srijith9862/screening-agent
```
(A GitHub sign-in window may pop up; sign in.) No winget? Download the ZIP instead: on the GitHub repo page → green **Code** button → **Download ZIP** → unzip into Documents.

### Step 2: Run the setup script
Open the `screening-agent` folder in File Explorer and **double-click `setup.bat`**. It:
1. Installs **Python 3.12** if you don't have it (via winget).
2. Downloads **ADB** (Android platform-tools) from Google, adds it to PATH, and puts its path in `config/settings.yaml`.
3. Downloads and launches the installers for two free virtual audio cables from vb-audio.com: **VB-CABLE** and **Hi-Fi Cable**. Windows asks "Allow changes?" → **Yes**, then click **Install Driver** / **Install** in each window and close it.
4. Installs the Python packages into `.venv`.
5. Opens the Anthropic and Deepgram key pages in your browser and asks you to paste each key into the window (typing is hidden). The keys are saved to `.env` on your laptop only.
   - **Anthropic** (console.anthropic.com/settings/keys): sign up → add a payment method under Billing → **Create Key** → copy.
   - **Deepgram** (console.deepgram.com): sign up (new accounts get free credit) → **API Keys** → **Create a New API Key** → copy.
6. Runs `screener check` to show what's left.

If Windows SmartScreen warns about `setup.bat`, click **More info → Run anyway**. You can run it again any time; finished steps are skipped.

**Restart the laptop** after the audio drivers are installed.

### Step 3: The phone (S24 Ultra)
1. **Settings → About phone → Software information** → tap **Build number** 7 times (enter your PIN). "Developer mode has been enabled" appears.
2. **Settings → Developer options** (at the bottom of Settings) → turn on **USB debugging** and **Stay awake**.
3. Connect the phone to the laptop with a USB cable. On the phone, tap **Allow** on "Allow USB debugging?" (tick **Always allow from this computer**).
4. Check it worked: in the `screening-agent` folder run `screener check`. "ADB devices" should list your phone.
5. While calling: keep the phone unlocked, plugged in, and don't use it for anything else.

### Step 4: Phone Link (so the laptop carries the call audio)
1. On the laptop open **Phone Link** (Start menu → `Phone Link`) → **Android**.
2. On the phone, swipe down twice and tap **Link to Windows** (or Settings → Connected devices → Link to Windows), then scan the QR code shown on the laptop.
3. In Phone Link, open the **Calls** tab → **Get started** → follow the Bluetooth pairing prompts on both devices (confirm the matching code).
4. Make a test call from Phone Link to any number. You should hear it through the laptop.

### Step 5: Route Phone Link's audio through the cables
The agent listens on one cable and speaks into the other:

| Phone Link setting | Set it to | Why |
|---|---|---|
| Output (speaker) | **CABLE Input (VB-Audio Virtual Cable)** | the candidate's voice goes to the agent |
| Input (microphone) | **Hi-Fi Cable Output (VB-Audio Hi-Fi Cable)** | the agent's voice goes into the call |

How:
1. Start a test call from Phone Link (Phone Link only shows up in the mixer while it's using audio).
2. Open **Settings → System → Sound → Volume mixer** (Windows 10: **Sound → App volume and device preferences**).
3. Find **Phone Link** in the apps list, expand it, and set **Output device** and **Input device** as in the table.
4. Hi-Fi Cable needs matching sample rates: press `Win+R`, type `mmsys.cpl`, Enter.
   - **Playback** tab → **Hi-Fi Cable Input** → Properties → Advanced → choose **48000 Hz** → OK.
   - **Recording** tab → **Hi-Fi Cable Output** → Properties → Advanced → choose **48000 Hz** → OK.
5. Want to listen in on calls? In the same window: **Recording** tab → **CABLE Output** → Properties → **Listen** tab → tick **Listen to this device** → pick your headphones.

> **Calls the agent dials (from the phone) use the Windows *default* devices**, not the Phone Link setting above. While the agent is calling, in `mmsys.cpl`:
> - **Playback** tab: right-click **CABLE Input** → **Set as Default Device**, and again → **Set as Default Communication Device**.
> - **Recording** tab: right-click **Hi-Fi Cable Output** → **Set as Default Device**, and again → **Set as Default Communication Device**.
> - To hear the call yourself: **Recording** tab → **CABLE Output** and **Hi-Fi Cable Output** → Properties → **Listen** → tick **Listen to this device** → playback through **Speakers (Realtek)** (pick it by name, not "Default").
>
> Switch the defaults back to your speakers and microphone when you're done calling.

If your device names differ, run `screener devices` and update the `audio:` section of `config/settings.yaml`.

### Step 6: Test the audio
With a test call to a friend running in Phone Link:
```bash
screener audio-test
```
Your friend should hear a beep, and the level bar should move when they talk. (`screener` is the `screener.bat` shortcut in this folder; run commands from the `screening-agent` folder.)

---

## 2. Before calling: fill in your details

- `config/faq.yaml`: replace any remaining `TODO`. Until then, the agent tells candidates "the HR team will get back to you" for those topics.
- `config/settings.yaml` → `script_values`: office schedule.
- `config/script.yaml`: the questions. Edit freely.
- Put your Naukri shortlist at `data/candidates.csv`. It needs a **`Phone`** column (`+919876543210` or a 10-digit number). An optional `Preferred_call_time` column is shown before each call. `Category` and `Rating_out_of_10` are deliberately **not** shown to the AI, so it isn't biased.

`data/` (except the sample) and `output/` are git-ignored: candidate data never goes to GitHub.

---

## 3. Practise first

**Hear the voice** on your laptop speakers. The agent is "Priya", using Deepgram's Indian English voice `flux-priya-en`:
```bash
screener voice-test
```
To try another voice: `screener voice-test --voice flux-meena-en`. To change it permanently, edit `deepgram.tts_voice` in `config/settings.yaml`, and `agent_name` if you want the agent to introduce itself by a different name.

**Simulated candidates** (only needs the Anthropic key). Claude plays six candidates against the agent (strong, vague, asks lots of questions, refuses recording, wants a call-back, has knockouts), then you get transcripts, scorecards and automatic checks (AI disclosure, recording consent, over-budget CTC flagged):
```bash
screener simulate
```
Run a subset with `--only curious,declines`. A full run costs roughly $2-4 in API usage.

**Text practice** (only needs the Anthropic key). You type as the candidate:
```bash
screener practice --name "Harshitha"
```

**Voice rehearsal** on your laptop mic. **Use headphones**, otherwise the agent hears itself:
```bash
screener rehearse --name "Harshitha"
```

Both write a scorecard to `output/` so you can check the scoring too.

---

## 4. Make calls

```bash
screener call
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
screener score output\calls\2026-11-07\harshitha-devulapalli-103015
```

### Retention
Recordings are personal data (DPDP Act 2023). Delete call folders older than `retention_days` (90 by default):
```bash
screener purge
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Agent starts talking during the caller tune / never starts | Run `screener adb-state` during a call. If you don't see `state: ACTIVE` lines once answered, set `phone.answer_detection: speech`. |
| Candidate can't hear the agent | Phone Link input device must be `Hi-Fi Cable Output`; `audio.call_out_device` must be `Hi-Fi Cable Input`; both at 48000 Hz. Re-run `audio-test`. |
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
