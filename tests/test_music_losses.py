import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.data import collate_states
from music_graph_dfm.diffusion.losses import music_structure_loss
from music_graph_dfm.diffusion.masking import coordinate_masks
from music_graph_dfm.diffusion.state_ops import batch_to_coords
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import empty_state


def _dummy_outputs_from_coords(coords: dict, vocab: int = 8) -> dict:
    out = {}
    for coord in COORD_ORDER:
        shape = coords[coord].shape
        out[coord] = {
            "lambda": torch.zeros((*shape, 1), dtype=torch.float32),
            "logits": torch.zeros((*shape, vocab), dtype=torch.float32),
        }
    return out


def test_music_structure_loss_fast_mode_skips_decoded_penalties():
    state = empty_state(num_spans=1, num_notes=1)
    state.span_attrs["key"] = [0]
    state.span_attrs["harm_root"] = [0]
    state.span_attrs["harm_quality"] = [1]
    state.span_attrs["meter"] = [4]
    state.span_attrs["section"] = [0]
    state.span_attrs["reg_center"] = [4]
    state.note_attrs["active"] = [1]
    state.note_attrs["pitch_token"] = [1]
    state.note_attrs["velocity"] = [8]
    state.note_attrs["role"] = [0]
    state.host = [1]
    state.template = [1]
    batch = collate_states([state])
    coords = batch_to_coords(batch)
    outputs = _dummy_outputs_from_coords(coords, vocab=12)
    masks = coordinate_masks(batch)

    rhythm = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8)
    rhythm.fit([(4, 0, 3, 0, 0)])
    pitch = PitchTokenCodec()
    compat = torch.tensor(pitch.compatibility_table(), dtype=torch.float32)

    loss = music_structure_loss(
        outputs=outputs,
        x_t=coords,
        batch=batch,
        masks=masks,
        rhythm_vocab=rhythm,
        pitch_codec=pitch,
        compat_table=compat,
        fast_music_loss_only=True,
    )
    assert float(loss["duplicate"].item()) == 0.0
    assert float(loss["voice_leading"].item()) == 0.0
    assert float(loss["repetition"].item()) == 0.0
    assert float(loss["full_decoded_executed"]) == 0.0
