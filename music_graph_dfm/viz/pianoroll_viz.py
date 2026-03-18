"""Piano-roll and optional audio rendering utilities."""

from __future__ import annotations

from pathlib import Path

from music_graph_dfm.data.fsntg import FSNTGState
from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.utils.midi import save_midi, to_pianoroll


def save_pianoroll(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    path: str | Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for piano-roll visualization") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roll = to_pianoroll(state, template_vocab, pitch_codec)

    plt.figure(figsize=(14, 6))
    plt.imshow(roll, origin="lower", aspect="auto", cmap="magma")
    plt.xlabel("time bins")
    plt.ylabel("MIDI pitch")
    plt.title("Decoded piano-roll")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def render_audio_with_fluidsynth(midi_path: str | Path, wav_path: str | Path, soundfont: str | Path) -> Path:
    try:
        import subprocess
    except Exception as exc:
        raise RuntimeError("subprocess unavailable") from exc

    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    soundfont = Path(soundfont)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fluidsynth",
        "-ni",
        str(soundfont),
        str(midi_path),
        "-F",
        str(wav_path),
        "-r",
        "44100",
    ]
    subprocess.check_call(cmd)
    return wav_path


def save_midi_and_roll(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    out_dir: str | Path,
    prefix: str,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = save_midi(out_dir / f"{prefix}.mid", state, template_vocab, pitch_codec)
    roll_path = save_pianoroll(state, template_vocab, pitch_codec, out_dir / f"{prefix}_pianoroll.png")
    return {
        "midi": str(midi_path),
        "pianoroll": str(roll_path),
    }
