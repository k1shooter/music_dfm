"""Graph visualization for FSNTG samples."""

from __future__ import annotations

from pathlib import Path

from music_graph_dfm.data.fsntg import FSNTGState


def save_graph_visualization(state: FSNTGState, path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as exc:
        raise RuntimeError("matplotlib and networkx are required for graph visualization") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    for j in range(state.num_spans):
        g.add_node(f"S{j}", kind="span")
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 1:
            g.add_node(f"N{i}", kind="note")

    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        for j, tpl in enumerate(state.e_ns[i]):
            if tpl != 0:
                g.add_edge(f"N{i}", f"S{j}", label=f"tpl:{tpl}")

    for j in range(state.num_spans):
        for k in range(state.num_spans):
            rel = state.e_ss[j][k]
            if rel != 0:
                g.add_edge(f"S{j}", f"S{k}", label=f"rel:{rel}")

    pos = {}
    for j in range(state.num_spans):
        pos[f"S{j}"] = (j, 1.0)
    active_note_ids = [i for i in range(state.num_notes) if int(state.note_attrs["active"][i]) == 1]
    for idx, i in enumerate(active_note_ids):
        pos[f"N{i}"] = (idx * max(1, state.num_spans) / max(1, len(active_note_ids)), 0.0)

    node_colors = ["#1f77b4" if g.nodes[n]["kind"] == "span" else "#ff7f0e" for n in g.nodes]

    plt.figure(figsize=(14, 6))
    nx.draw_networkx(g, pos=pos, node_color=node_colors, node_size=600, with_labels=True, font_size=8, arrows=True)
    edge_labels = nx.get_edge_attributes(g, "label")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
