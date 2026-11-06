"""Command line: python -m screener <command>."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import storage
from .candidates import Candidate, load_candidates
from .config import Settings, TimeWindow, load_env, load_settings, require_env

DEFAULT_CSV = "data/candidates.csv"


# --- helpers ---------------------------------------------------------------


def _pick(candidates: list[Candidate], name: str | None) -> list[Candidate]:
    if not name:
        return candidates
    matches = [c for c in candidates if name.lower() in c.name.lower()]
    if not matches:
        raise SystemExit(f"No candidate matching '{name}'.")
    return matches


def _load(args: argparse.Namespace, settings: Settings) -> list[Candidate]:
    path = Path(args.candidates)
    if not path.exists():
        raise SystemExit(f"Candidate file not found: {path}. Put your Naukri CSV there or pass --candidates.")
    return _pick(load_candidates(path, settings.get("phone.default_country_code", "+91")), args.name)


def _sample_candidate() -> Candidate:
    return Candidate(
        name="Test Candidate",
        fields={
            "hr_experience": "3 years",
            "e2e_recruitment_experience": "2 years",
            "expected_ctc": "5 lakhs",
            "notice_period": "30 days",
            "office_ok": "Yes",
        },
    )


def _within_calling_hours(settings: Settings) -> bool:
    window = TimeWindow.parse(settings.get("call.calling_hours", "10:00-19:00"))
    now = datetime.now(ZoneInfo(settings.get("call.timezone", "Asia/Kolkata")))
    return window.contains(now.hour, now.minute)


def _warn_todos(settings: Settings) -> None:
    for warning in settings.todo_warnings():
        print(f"  ! {warning}")


async def _finish(settings: Settings, candidate: Candidate, result, call_dir: Path, score: bool = True) -> dict:
    """Saves what consent allows, runs the scorecard and appends to results.csv."""
    from .scorer import format_transcript, score_call, scorecard_markdown

    row = {
        "called_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": candidate.name,
        "phone": candidate.phone,
        "folder": str(call_dir),
        "recording_consent": result.consent,
        "callback_request": result.callback_request,
    }
    meta = {
        "candidate": candidate.name,
        "outcome": result.outcome,
        "ended_by": result.ended_by,
        "duration_s": result.duration_s,
        "recording_consent": result.consent,
        "callback_request": result.callback_request,
    }

    if result.outcome != "connected":
        row["outcome"] = result.outcome
    elif result.consent != "yes":
        # No consent: keep no audio and no transcript, only the outcome.
        if result.recorder:
            result.recorder.discard()
        row["outcome"] = "declined_recording" if result.consent == "no" else "ended_before_consent"
        if result.callback_request:
            row["outcome"] = "call_back_later"
    else:
        if result.recorder and result.recorder.save(call_dir / "recording.wav"):
            meta["recording"] = "recording.wav (left: candidate, right: agent)"
        storage.write_json(call_dir / "transcript.json", result.transcript)
        meta["agent_notes"] = result.notes
        if score:
            print("  Writing scorecard...", flush=True)
            transcript = format_transcript(result.transcript, settings.agent_name)
            try:
                card = await score_call(settings, candidate, transcript, result.notes)
            except Exception as exc:  # noqa: BLE001 - keep the call data even if scoring fails
                print(f"  ! Scoring failed: {exc}. Run `python -m screener score {call_dir}` later.")
                row["outcome"] = "completed_unscored"
            else:
                storage.write_json(call_dir / "scorecard.json", card)
                (call_dir / "scorecard.md").write_text(scorecard_markdown(candidate, card), encoding="utf-8")
                row.update(storage.scorecard_row(card))
                print(f"  Recommendation: {card['recommendation'].upper()} - {card['recommendation_reason']}")
        else:
            row["outcome"] = "completed_unscored"

    storage.write_json(call_dir / "call.json", meta)
    storage.append_result(settings, row)
    print(f"  Outcome: {row.get('outcome')}  ->  {call_dir}")
    return row


# --- commands ----------------------------------------------------------------


def cmd_devices(args: argparse.Namespace, settings: Settings) -> None:
    from .audio import list_devices

    print(list_devices())
    print(f"\nConfigured: call_in_device='{settings.get('audio.call_in_device')}', "
          f"call_out_device='{settings.get('audio.call_out_device')}'")


def cmd_check(args: argparse.Namespace, settings: Settings) -> None:
    import os

    from .audio import find_device
    from .phone import Phone, PhoneError

    ok = True
    print("Config:")
    _warn_todos(settings)
    if not settings.todo_warnings():
        print("  ok, no TODOs left")
    print("API keys:")
    for key in ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY"):
        present = bool(os.environ.get(key))
        ok &= present
        print(f"  {key}: {'set' if present else 'MISSING (add it to .env)'}")
    print("Audio:")
    for label, key, kind in (("call in", "audio.call_in_device", "input"), ("call out", "audio.call_out_device", "output")):
        try:
            index = find_device(settings.get(key), kind)
            print(f"  {label}: device #{index} matches '{settings.get(key)}'")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  {label}: {exc}")
    print("Phone:")
    phone = Phone(settings.get("phone.adb_path", "adb"), settings.get("phone.device_serial"))
    try:
        devices = phone.connected_devices()
        print(f"  ADB devices: {devices or 'none - connect the phone and allow USB debugging'}")
        ok &= bool(devices)
    except PhoneError as exc:
        ok = False
        print(f"  {exc}")
    print("Calling hours:", "inside window now" if _within_calling_hours(settings) else "outside window now")
    print("\nAll good." if ok else "\nFix the items above before calling.")


def cmd_adb_state(args: argparse.Namespace, settings: Settings) -> None:
    from .phone import Phone

    print(Phone(settings.get("phone.adb_path", "adb"), settings.get("phone.device_serial")).debug_state())


def cmd_transfer_test(args: argparse.Namespace, settings: Settings) -> None:
    """During a call started on the phone: clicks Phone Link's 'Transfer to laptop' popup."""
    from . import phonelink

    print("Looking for Phone Link's Transfer popup for 20 seconds...")
    clicked, details = asyncio.run(phonelink.click_transfer(timeout_s=20))
    print(details if details else ("Clicked." if clicked else "Not found."))


def cmd_bt_check(args: argparse.Namespace, settings: Settings) -> None:
    """During a call: shows the in-call Bluetooth button the agent would tap (and taps it with --tap)."""
    from .phone import BLUETOOTH_PATTERN, Phone

    phone = Phone(settings.get("phone.adb_path", "adb"), settings.get("phone.device_serial"))

    async def run() -> None:
        nodes = await phone.screen_nodes()
        matches = [n for n in nodes if BLUETOOTH_PATTERN.search(n.label)]
        print(f"Read {len(nodes)} labelled items from the phone screen.")
        if not matches:
            print("No 'Bluetooth' button found. Start a call and keep the call screen open (phone unlocked).")
            return
        for n in matches:
            print(f"  '{n.label}'  at {n.center}  clickable={n.clickable}  selected/on={n.selected}")
        if args.tap:
            print(await phone.tap_bluetooth(str(settings.get("phone.bluetooth_device_name") or "")))

    asyncio.run(run())


def _say_into_call(engine, text: str) -> None:
    """Speaks `text` into the call with the agent's voice (falls back to a beep)."""
    import os

    from .audio import tone
    from .speech import TTS_RATE, TextToSpeech

    key = os.environ.get("DEEPGRAM_API_KEY", "")
    audio = None
    if key:
        async def synth():
            tts = TextToSpeech(key, "flux-priya-en")
            try:
                return await tts.synthesize(text)
            finally:
                await tts.close()

        try:
            audio = asyncio.run(synth())
        except Exception as exc:  # noqa: BLE001 - a beep is good enough for the test
            print(f"  (voice unavailable: {exc})")
    if audio is None:
        engine.play(tone(1.0, engine.out_rate), engine.out_rate)
    else:
        engine.play(audio, TTS_RATE)
    while not engine.player.is_idle():
        time.sleep(0.05)


def _measure(engine, seconds: float, label: str) -> float:
    """Shows a live level bar for `seconds` and returns the average level."""
    levels = []
    end = time.time() + seconds
    while time.time() < end:
        levels.append(engine.input_level)
        bar = "#" * min(50, int(engine.input_level * 400))
        print(f"\r  {label:<8} {bar:<50} {int(end - time.time()) + 1:2d}s", end="", flush=True)
        time.sleep(0.1)
    print()
    return sum(levels) / max(1, len(levels))


def cmd_audio_test(args: argparse.Namespace, settings: Settings) -> None:
    """Checks both directions of call audio during a Phone Link call to a friend."""
    from .audio import AudioEngine, Recorder

    engine = AudioEngine(settings.get("audio.call_in_device"), settings.get("audio.call_out_device"), lambda _b: None, Recorder())
    print("Before running this, start a Phone Link call to a friend and keep it going.")
    print(f"  Agent speaks into : {engine.out_name}")
    print(f"  Agent listens to  : {engine.in_name}\n")
    engine.start()
    try:
        print("1) The agent is telling your friend what to do (they should hear it)...")
        _say_into_call(engine, "Hi, this is a sound check. Please stay completely quiet for five seconds.")
        quiet = _measure(engine, 5, "quiet")
        _say_into_call(engine, "Thank you. Now please count slowly from one to ten.")
        talking = _measure(engine, 7, "talking")
        _say_into_call(engine, "Thanks, that's all.")
    finally:
        engine.stop()

    print(f"\nAverage level  quiet: {quiet:.4f}   talking: {talking:.4f}")
    if talking > 0.003 and talking > quiet * 2.5:
        print("PASS: the agent can hear your friend. If your friend also heard the instructions, routing is done.")
    elif talking <= 0.003 and quiet <= 0.003:
        print("FAIL: no call audio is reaching the agent. Phone Link's speaker isn't going to 'CABLE Input'.")
        print("  Fix: Win+R > mmsys.cpl > Playback > right-click 'CABLE Input' > Set as Default Communication Device,")
        print("       then hang up, call again, and re-run this test.")
    else:
        print("UNCLEAR: audio arrives but talking wasn't clearly louder than silence.")
        print("  Check your friend counted out loud, the line wasn't noisy, then run the test again.")


def cmd_voice_test(args: argparse.Namespace, settings: Settings) -> None:
    """Plays the agent's voice on your laptop speakers."""
    from .audio import _sd
    from .speech import TTS_RATE, TextToSpeech

    key = require_env("DEEPGRAM_API_KEY")
    voice = args.voice or settings.get("deepgram.tts_voice", "flux-priya-en")
    text = args.text or (
        f"Hi, am I speaking with Harshitha? This is {settings.agent_name}, an AI assistant calling on behalf of "
        f"{settings.script.get('company', 'the company')} about your application for the HR Executive role. "
        "Is now a good time to talk for about twelve minutes?"
    )
    if args.compare:
        levels = [0, 1, 2]
    elif args.expressivity is not None:
        levels = [args.expressivity]
    else:
        levels = [int(settings.get("deepgram.tts_expressivity", 0))]

    async def synth(level: int):
        tts = TextToSpeech(key, voice, level)
        try:
            return await tts.synthesize(text)
        finally:
            await tts.close()

    sd = _sd()
    for level in levels:
        print(f"Voice: {voice}, expressivity {level}", flush=True)
        sd.play(asyncio.run(synth(level)), TTS_RATE)
        sd.wait()
        time.sleep(0.6)
    if len(levels) > 1 or args.expressivity is not None:
        print("To keep a level, set deepgram.tts_expressivity in config/settings.yaml.")


def cmd_practice(args: argparse.Namespace, settings: Settings) -> None:
    """Text-only rehearsal: you type as the candidate."""
    from .brain import Brain
    from .call_session import CallResult

    require_env("ANTHROPIC_API_KEY")
    candidate = _load(args, settings)[0] if Path(args.candidates).exists() else _sample_candidate()
    _warn_todos(settings)
    print(f"Practice call as {candidate.name}. Type what the candidate says. /quit to stop.\n")

    async def run() -> None:
        brain = Brain(settings, candidate)
        transcript = []
        t0 = time.monotonic()

        def log(speaker: str, text: str) -> None:
            transcript.append({"t": round(time.monotonic() - t0, 1), "speaker": speaker, "text": text})

        hello = input(f"{candidate.first_name}: ").strip() or "Hello?"
        log("candidate", hello)
        message = f'[Call connected. The candidate answered and said: "{hello}"]'
        while True:
            turn = await brain.respond(message)
            print(f"\n{settings.agent_name}: {turn.say}")
            if turn.note:
                print(f"   (note: {turn.note})")
            log("agent", turn.say)
            if turn.end_call:
                break
            reply = input(f"\n{candidate.first_name}: ").strip()
            if reply == "/quit":
                break
            message = reply or "[The candidate has been silent for 10 seconds. Check if they can hear you.]"
            if reply:
                log("candidate", reply)
        result = CallResult(
            outcome="connected",
            consent=brain.consent,
            transcript=transcript,
            notes=brain.notes,
            callback_request=brain.callback_request,
            duration_s=round(time.monotonic() - t0, 1),
        )
        call_dir = storage.new_call_dir(settings, candidate)
        await _finish(settings, candidate, result, call_dir, score=not args.no_score)

    asyncio.run(run())


def cmd_rehearse(args: argparse.Namespace, settings: Settings) -> None:
    """Voice rehearsal on your laptop mic and speakers (use headphones), no phone."""
    from .call_session import CallSession

    require_env("ANTHROPIC_API_KEY")
    key = require_env("DEEPGRAM_API_KEY")
    candidate = _load(args, settings)[0] if Path(args.candidates).exists() else _sample_candidate()
    print("Use headphones, otherwise the agent hears itself through your microphone.")

    async def run() -> None:
        session = CallSession(settings, candidate, None, key, args.mic, args.speaker)
        result = await session.run()
        await _finish(settings, candidate, result, storage.new_call_dir(settings, candidate), score=not args.no_score)

    asyncio.run(run())


def cmd_call(args: argparse.Namespace, settings: Settings) -> None:
    from .call_session import CallSession
    from .phone import Phone

    require_env("ANTHROPIC_API_KEY")
    key = require_env("DEEPGRAM_API_KEY")
    candidates = _load(args, settings)
    phone = Phone(settings.get("phone.adb_path", "adb"), settings.get("phone.device_serial"))
    if not phone.connected_devices():
        raise SystemExit("No phone found over ADB. Connect it by USB and allow USB debugging.")
    _warn_todos(settings)
    if settings.todo_warnings() and not args.yes:
        if input("There are TODOs in the config (above). Call anyway? [y/N] ").strip().lower() != "y":
            return

    done = set() if args.redo else storage.already_screened(settings)
    gap = int(settings.get("call.gap_between_calls_s", 90))
    confirm = settings.get("call.confirm_before_each_call", True) and not args.yes
    queue = [c for c in candidates if c.phone and c.phone not in done]
    skipped_no_phone = [c.name for c in candidates if not c.phone]
    if skipped_no_phone:
        print(f"Skipping (no valid phone number): {', '.join(skipped_no_phone)}")
    print(f"{len(queue)} candidate(s) to call.\n")

    async def run() -> None:
        for index, candidate in enumerate(queue):
            if not _within_calling_hours(settings):
                print(f"Outside calling hours ({settings.get('call.calling_hours')} IST). Stopping.")
                return
            print(f"[{index + 1}/{len(queue)}] {candidate.name}  {candidate.phone}"
                  + (f"  (prefers: {candidate.fields['preferred_call_time']})" if candidate.fields.get("preferred_call_time") else ""))
            if confirm:
                answer = (await asyncio.to_thread(input, "  Call now? [Y]es / [s]kip / [q]uit: ")).strip().lower()
                if answer == "q":
                    return
                if answer == "s":
                    continue
            call_dir = storage.new_call_dir(settings, candidate)
            session = CallSession(
                settings, candidate, phone, key,
                settings.get("audio.call_in_device"), settings.get("audio.call_out_device"),
            )
            result = await session.run()
            await _finish(settings, candidate, result, call_dir)
            if index < len(queue) - 1:
                print(f"  Waiting {gap}s before the next call...\n")
                await asyncio.sleep(gap)

    asyncio.run(run())


def cmd_simulate(args: argparse.Namespace, settings: Settings) -> None:
    """Claude plays several candidates; the agent screens each; prints a report."""
    from .simulate import PERSONAS, checks, run_all

    require_env("ANTHROPIC_API_KEY")
    keys = [k.strip() for k in args.only.split(",")] if args.only else None
    if keys:
        unknown = set(keys) - {p.key for p in PERSONAS}
        if unknown:
            raise SystemExit(f"Unknown persona(s): {', '.join(unknown)}. Choose from: {', '.join(p.key for p in PERSONAS)}")
    names = ", ".join(keys or [p.key for p in PERSONAS])
    print(f"Running simulated calls ({names}). This takes a few minutes...", flush=True)
    results, folder = asyncio.run(run_all(settings, keys, score=not args.no_score))
    print()
    for r in results:
        issues = checks(r)
        rec = r.card["recommendation"].upper() if r.card else "-"
        status = "OK" if not issues else "CHECK: " + "; ".join(issues)
        print(f"  {r.persona.key:<9} consent={r.consent:<8} recommendation={rec:<8} turns={len(r.transcript):<3} {status}")
    print(f"\nFull transcripts and scorecards: {folder}")
    print(f"Open {folder / 'SUMMARY.md'} first.")


def cmd_score(args: argparse.Namespace, settings: Settings) -> None:
    from .scorer import format_transcript, score_call, scorecard_markdown

    require_env("ANTHROPIC_API_KEY")
    call_dir = Path(args.folder)
    transcript = json.loads((call_dir / "transcript.json").read_text(encoding="utf-8"))
    meta = json.loads((call_dir / "call.json").read_text(encoding="utf-8"))
    candidate = Candidate(name=meta.get("candidate", "Candidate"))
    if Path(args.candidates).exists():
        matches = [c for c in load_candidates(Path(args.candidates)) if c.name == candidate.name]
        candidate = matches[0] if matches else candidate
    card = asyncio.run(score_call(settings, candidate, format_transcript(transcript, settings.agent_name), meta.get("agent_notes", [])))
    storage.write_json(call_dir / "scorecard.json", card)
    (call_dir / "scorecard.md").write_text(scorecard_markdown(candidate, card), encoding="utf-8")
    print(f"Saved scorecard to {call_dir / 'scorecard.md'}")


def cmd_purge(args: argparse.Namespace, settings: Settings) -> None:
    days = args.days if args.days is not None else int(settings.get("storage.retention_days", 90))
    removed = storage.purge(settings, days)
    print(f"Deleted {len(removed)} call folder(s) older than {days} days.")


def main(argv: list[str] | None = None) -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="python -m screener", description="AI phone screening agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_candidates(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--candidates", default=DEFAULT_CSV, help=f"CSV file (default {DEFAULT_CSV})")
        p.add_argument("--name", help="only candidates whose name contains this")
        return p

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)
    sub.add_parser("check", help="check config, keys, audio and phone").set_defaults(func=cmd_check)
    sub.add_parser("adb-state", help="show the phone's call state (debugging)").set_defaults(func=cmd_adb_state)
    sub.add_parser("transfer-test", help="during a call on the phone: click Phone Link's Transfer popup").set_defaults(
        func=cmd_transfer_test
    )
    p = sub.add_parser("bt-check", help="during a call: show (or --tap) the phone's Bluetooth call button")
    p.add_argument("--tap", action="store_true", help="also tap it")
    p.set_defaults(func=cmd_bt_check)
    p = sub.add_parser("voice-test", help="hear the agent's voice on your laptop speakers")
    p.add_argument("text", nargs="?", help="what to say (default: the opening line)")
    p.add_argument("--voice", help="try another voice, e.g. flux-meena-en")
    p.add_argument("--expressivity", type=int, choices=[-2, -1, 0, 1, 2], help="-2 calm ... 2 animated")
    p.add_argument("--compare", action="store_true", help="play expressivity 0, 1 and 2 one after another")
    p.set_defaults(func=cmd_voice_test)
    sub.add_parser("audio-test", help="test call audio routing during a Phone Link call").set_defaults(func=cmd_audio_test)
    p = with_candidates(sub.add_parser("practice", help="text rehearsal: you type as the candidate"))
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(func=cmd_practice)
    p = with_candidates(sub.add_parser("rehearse", help="voice rehearsal with laptop mic/headphones"))
    p.add_argument("--mic", help="input device name (default: system default)")
    p.add_argument("--speaker", help="output device name (default: system default)")
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(func=cmd_rehearse)
    p = with_candidates(sub.add_parser("call", help="call candidates through the phone"))
    p.add_argument("--yes", action="store_true", help="don't ask before each call")
    p.add_argument("--redo", action="store_true", help="also call candidates already screened")
    p.set_defaults(func=cmd_call)
    p = sub.add_parser("simulate", help="test the agent against simulated candidates (no phone)")
    p.add_argument("--only", help="comma-separated personas: strong,vague,curious,declines,callback,knockout")
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser("score", help="re-run the scorecard for a call folder")
    p.add_argument("folder")
    p.add_argument("--candidates", default=DEFAULT_CSV)
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("purge", help="delete call folders older than the retention period")
    p.add_argument("--days", type=int)
    p.set_defaults(func=cmd_purge)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ₹ and names on the Windows console
    args.func(args, load_settings())
