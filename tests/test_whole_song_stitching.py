from music_graph_dfm.representation.state import empty_state
from music_graph_dfm.whole_song.generation import stitch_segments_baseline


def test_whole_song_stitching_bookkeeping():
    a = empty_state(num_spans=2, num_notes=1)
    a.note_attrs["active"] = [1]
    a.host = [1]
    a.template = [1]

    b = empty_state(num_spans=3, num_notes=2)
    b.note_attrs["active"] = [1, 1]
    b.host = [1, 3]
    b.template = [2, 3]

    stitched = stitch_segments_baseline([a, b])
    assert stitched.num_spans == 5
    assert stitched.num_notes == 3
    assert stitched.host[1] == 3  # shifted by first segment's 2 spans
    assert stitched.host[2] == 5
