"""Whole-song generation modes."""

from music_graph_dfm.whole_song.generation import (
    build_long_context_template,
    generate_whole_song,
    stitch_segments_baseline,
)

__all__ = ["build_long_context_template", "generate_whole_song", "stitch_segments_baseline"]
