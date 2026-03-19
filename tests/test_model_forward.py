import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.data import collate_states, infer_vocab_sizes
from music_graph_dfm.models import FSNTGV2HeteroTransformer, ModelConfig
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import empty_state


def _state():
    st = empty_state(num_spans=2, num_notes=3)
    st.span_attrs["key"] = [0, 2]
    st.span_attrs["harm_root"] = [0, 7]
    st.span_attrs["harm_quality"] = [1, 3]
    st.span_attrs["harm_function"] = [1, 3]
    st.span_attrs["meter"] = [4, 4]
    st.span_attrs["section"] = [0, 1]
    st.span_attrs["reg_center"] = [4, 4]
    st.note_attrs["active"] = [1, 1, 0]
    st.note_attrs["pitch_token"] = [1, 2, 0]
    st.note_attrs["velocity"] = [8, 9, 0]
    st.note_attrs["role"] = [0, 1, 0]
    st.host = [1, 2, 0]
    st.template = [1, 2, 0]
    st.e_ss[0][1] = 1
    return st


def _template_spec(vocab_size: int):
    rhythm = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8)
    rhythm.fit([(4, 0, 3, 0, 0), (4, 2, 3, 0, 0), (4, 4, 2, 0, 0)])
    return {
        "onset_bin": [rhythm.decode(i).onset_bin for i in range(vocab_size)],
        "duration_class": [rhythm.decode(i).duration_class for i in range(vocab_size)],
        "tie_flag": [rhythm.decode(i).tie_flag for i in range(vocab_size)],
        "extension_class": [rhythm.decode(i).extension_class for i in range(vocab_size)],
        "duration_ticks": list(rhythm.duration_ticks),
        "onset_bins": int(rhythm.onset_bins),
        "tie_extension_fraction": float(rhythm.tie_extension_fraction),
    }


@pytest.mark.parametrize("model_kind", ["early_sum", "late_fusion"])
def test_model_forward_shapes(model_kind: str):
    st = _state()
    vocab = infer_vocab_sizes([st])
    model = FSNTGV2HeteroTransformer(
        vocab_sizes=vocab,
        cfg=ModelConfig(hidden_dim=16, num_layers=2, num_heads=2, dropout=0.0),
        template_spec=_template_spec(max(2, int(vocab["note.template"]))),
        model_kind=model_kind,
    )
    batch = collate_states([st])
    outputs = model(batch, torch.tensor(0.5))
    for coord in COORD_ORDER:
        assert coord in outputs
        assert "lambda" in outputs[coord]
        assert "logits" in outputs[coord]

    edit_outputs = model.forward_edit(batch, torch.tensor(0.5))
    for key in [
        "lambda_type",
        "type_logits",
        "note_logits",
        "host_logits",
        "template_logits",
        "pitch_logits",
        "velocity_logits",
        "role_logits",
        "span_src_logits",
        "span_dst_logits",
        "insert_host_logits",
        "insert_template_logits",
        "insert_pitch_logits",
        "insert_velocity_logits",
        "insert_role_logits",
        "span_rel_logits",
    ]:
        assert key in edit_outputs
