"""Masked losses for coordinate-wise DFM objectives."""

from __future__ import annotations

from typing import Dict

from music_graph_dfm.constants import COORD_ORDER, GRAPH_KERNEL_APPROX_COORDS
from music_graph_dfm.data.tensor_codec import coords_to_states
from music_graph_dfm.diffusion.masking import enforce_state_constraints
from music_graph_dfm.diffusion.paths import graph_kernel_target_distribution, graph_kernel_target_rate_approximation
from music_graph_dfm.representation.state import reconstruct_aux_graph


def _masked_mean(values, mask):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_masked_mean requires torch") from exc

    num = (values * mask).sum()
    den = mask.sum().clamp(min=1.0)
    return num / den


def _target_distribution(coord: str, x1, path_meta: dict, vocab_size: int):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_target_distribution requires torch") from exc

    path_type = path_meta.get("path_type", "mixture")
    kernels = path_meta.get("graph_kernels", {})
    if path_type == "graph_kernel" and coord in kernels and coord in set(GRAPH_KERNEL_APPROX_COORDS):
        return graph_kernel_target_distribution(x1, kernels[coord])
    x1 = x1.clamp(min=0, max=vocab_size - 1)
    return torch.nn.functional.one_hot(x1, num_classes=vocab_size).to(torch.float32)


def rate_matching_loss(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    x_1: Dict[str, "torch.Tensor"],
    xt_is_x0: Dict[str, "torch.Tensor"],
    eta: Dict[str, float],
    masks: Dict[str, "torch.Tensor"],
    path_meta: dict,
    eps: float = 1e-8,
):
    """Poisson rate matching on off-diagonal rates only."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("rate_matching_loss requires torch") from exc

    total = torch.tensor(0.0, device=next(iter(x_t.values())).device)
    for coord in COORD_ORDER:
        out = outputs[coord]
        lam = torch.nn.functional.softplus(out["lambda"]).squeeze(-1)
        logits = out["logits"]
        pi = torch.softmax(logits, dim=-1)

        xt = x_t[coord]
        x1 = x_1[coord]
        mask = masks[coord].to(torch.float32)
        indicator = xt_is_x0[coord].to(torch.float32)

        vocab = pi.shape[-1]
        current = torch.nn.functional.one_hot(xt.clamp(min=0, max=vocab - 1), num_classes=vocab).to(torch.float32)
        pred_offdiag = lam.unsqueeze(-1) * pi * (1.0 - current)

        if (
            path_meta.get("path_type", "mixture") == "graph_kernel"
            and coord in path_meta.get("graph_kernels", {})
            and coord in set(GRAPH_KERNEL_APPROX_COORDS)
        ):
            approx = graph_kernel_target_rate_approximation(xt, x1, eta=float(eta[coord]), kernel=path_meta["graph_kernels"][coord])
            target_rates = indicator.unsqueeze(-1) * approx
        else:
            target_dist = _target_distribution(coord, x1, path_meta, vocab)
            target_rates = indicator.unsqueeze(-1) * float(eta[coord]) * target_dist * (1.0 - current)

        m = mask.unsqueeze(-1)
        nll = pred_offdiag - target_rates * torch.log(pred_offdiag + eps)
        total = total + _masked_mean(nll.sum(dim=-1), mask)
    return total


def auxiliary_denoising_loss(
    outputs: Dict[str, dict],
    x_1: Dict[str, "torch.Tensor"],
    masks: Dict[str, "torch.Tensor"],
):
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("auxiliary_denoising_loss requires torch") from exc

    losses = []
    for coord in COORD_ORDER:
        logits = outputs[coord]["logits"]
        target = x_1[coord]
        vocab = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1), reduction="none").reshape(target.shape)
        losses.append(_masked_mean(ce, masks[coord].to(torch.float32)))
    return torch.stack(losses).mean()


def host_uniqueness_penalty(coords: Dict[str, "torch.Tensor"], masks: Dict[str, "torch.Tensor"]):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("host_uniqueness_penalty requires torch") from exc

    active = coords["note.active"].to(torch.float32)
    has_host = (coords["note.host"] > 0).to(torch.float32)
    diff = (active - has_host).abs()
    return _masked_mean(diff, masks["note.active"].to(torch.float32))


def host_uniqueness_penalty_from_outputs(outputs: Dict[str, dict], masks: Dict[str, "torch.Tensor"]):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("host_uniqueness_penalty_from_outputs requires torch") from exc

    active_prob = torch.softmax(outputs["note.active"]["logits"], dim=-1)
    if active_prob.shape[-1] > 1:
        active_on = 1.0 - active_prob[..., 0]
    else:
        active_on = active_prob[..., 0]

    host_prob = torch.softmax(outputs["note.host"]["logits"], dim=-1)
    has_host = 1.0 - host_prob[..., 0]
    diff = (active_on - has_host).abs()
    return _masked_mean(diff, masks["note.active"].to(torch.float32))


def harmonic_compatibility_penalty_from_outputs(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    masks: Dict[str, "torch.Tensor"],
    compat_table: "torch.Tensor" | None,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("harmonic_compatibility_penalty_from_outputs requires torch") from exc

    if compat_table is None:
        return torch.tensor(0.0, device=next(iter(x_t.values())).device)

    host_prob = torch.softmax(outputs["note.host"]["logits"], dim=-1)  # [B,N,H]
    pitch_prob = torch.softmax(outputs["note.pitch_token"]["logits"], dim=-1)  # [B,N,T]
    note_mask = masks["note.active"].to(torch.float32)

    bsz, n, h_vocab = host_prob.shape
    pitch_vocab = pitch_prob.shape[-1]
    span_count = x_t["span.key"].shape[1]
    if h_vocab <= 1 or span_count <= 0:
        return torch.tensor(0.0, device=host_prob.device)

    # Build compatibility rows for each host category in a batched way.
    compat_by_host = torch.zeros((bsz, h_vocab, pitch_vocab), dtype=torch.float32, device=host_prob.device)
    for host in range(1, h_vocab):
        span_idx = host - 1
        if span_idx >= span_count:
            continue
        key = x_t["span.key"][:, span_idx].clamp(min=0, max=compat_table.shape[0] - 1).to(torch.long)
        harm_root = x_t["span.harm_root"][:, span_idx].clamp(min=0, max=compat_table.shape[1] - 1).to(torch.long)
        if compat_table.dim() == 3:
            compat_row = compat_table[key, harm_root, :pitch_vocab]
        elif compat_table.dim() == 4:
            harm_quality = x_t["span.harm_quality"][:, span_idx].clamp(min=0, max=compat_table.shape[2] - 1).to(torch.long)
            compat_row = compat_table[key, harm_root, harm_quality, :pitch_vocab]
        else:
            harm_quality = x_t["span.harm_quality"][:, span_idx].clamp(min=0, max=compat_table.shape[2] - 1).to(torch.long)
            if "span.harm_function" in x_t:
                harm_function = x_t["span.harm_function"][:, span_idx].clamp(min=0, max=compat_table.shape[3] - 1).to(torch.long)
            else:
                harm_function = torch.zeros_like(harm_quality)
            compat_row = compat_table[key, harm_root, harm_quality, harm_function, :pitch_vocab]
        compat_by_host[:, host, :] = compat_row

    compat_note_host = torch.einsum("bht,bnt->bnh", compat_by_host, pitch_prob)
    host_nonzero = host_prob[:, :, 1:]
    compat_nonzero = compat_note_host[:, :, 1:]
    host_mass = host_nonzero.sum(dim=-1).clamp(min=1e-8)
    exp_compat = (host_nonzero * compat_nonzero).sum(dim=-1) / host_mass
    penalty = (1.0 - exp_compat) * note_mask
    return penalty.sum() / note_mask.sum().clamp(min=1.0)


def _argmax_coords_from_outputs(outputs: Dict[str, dict], x_t: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    coords: Dict[str, "torch.Tensor"] = {}
    for coord in COORD_ORDER:
        logits = outputs[coord]["logits"]
        coords[coord] = logits.argmax(dim=-1)
    return coords


def _decoded_structure_penalties(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    batch: dict,
    rhythm_vocab,
    pitch_codec,
    subsample_notes: int = 0,
    subsample_pairs: int = 0,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_decoded_structure_penalties requires torch") from exc

    pred_coords = _argmax_coords_from_outputs(outputs, x_t)
    pred_coords = enforce_state_constraints(pred_coords, batch)
    states = coords_to_states(pred_coords, batch)

    dup_vals = []
    vl_vals = []
    rep_vals = []
    for state in states:
        notes = state.decode_notes(rhythm_vocab, pitch_codec)
        if subsample_notes > 0 and len(notes) > subsample_notes:
            notes = sorted(notes, key=lambda n: (n.onset_tick, n.note_idx))[:subsample_notes]

        seen = set()
        duplicates = 0
        for note in notes:
            key = (note.host_span, note.onset_tick, note.duration_tick, note.pitch, note.role)
            if key in seen:
                duplicates += 1
            seen.add(key)
        dup_vals.append(float(duplicates) / max(1, len(notes)))

        aux = reconstruct_aux_graph(state, rhythm_vocab, pitch_codec)
        note_lookup = {n.note_idx: n for n in notes}
        total = 0
        bad = 0
        seq_pairs = list(aux.sequential_same_role)
        if subsample_pairs > 0 and len(seq_pairs) > subsample_pairs:
            seq_pairs = seq_pairs[:subsample_pairs]
        for src, dst in seq_pairs:
            if src not in note_lookup or dst not in note_lookup:
                continue
            total += 1
            if abs(int(note_lookup[dst].pitch) - int(note_lookup[src].pitch)) > 12:
                bad += 1
        vl_vals.append(float(bad) / max(1, total))

        pairs = []
        for i in range(state.num_spans):
            for j in range(state.num_spans):
                rel = int(state.e_ss[i][j])
                if rel in {2, 3}:  # repeat / variation
                    pairs.append((i, j))
        if not pairs:
            rep_vals.append(0.0)
        else:
            if subsample_pairs > 0 and len(pairs) > subsample_pairs:
                pairs = pairs[:subsample_pairs]
            mismatch = sum(
                1
                for i, j in pairs
                if int(state.span_attrs["section"][i]) != int(state.span_attrs["section"][j])
            )
            rep_vals.append(float(mismatch) / len(pairs))

    return {
        "duplicate": torch.tensor(dup_vals, device=x_t["note.active"].device).mean() if dup_vals else torch.tensor(0.0, device=x_t["note.active"].device),
        "voice_leading": torch.tensor(vl_vals, device=x_t["note.active"].device).mean() if vl_vals else torch.tensor(0.0, device=x_t["note.active"].device),
        "repetition": torch.tensor(rep_vals, device=x_t["note.active"].device).mean() if rep_vals else torch.tensor(0.0, device=x_t["note.active"].device),
    }


def music_structure_loss(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    batch: dict,
    masks: Dict[str, "torch.Tensor"],
    rhythm_vocab,
    pitch_codec,
    compat_table: "torch.Tensor" | None = None,
    fast_music_loss_only: bool = False,
    structure_loss_subsample_notes: int = 0,
    structure_loss_subsample_pairs: int = 0,
):
    host = host_uniqueness_penalty_from_outputs(outputs, masks)
    harm = harmonic_compatibility_penalty_from_outputs(outputs, x_t, masks, compat_table)
    if fast_music_loss_only:
        decoded = {
            "duplicate": host.new_tensor(0.0),
            "voice_leading": host.new_tensor(0.0),
            "repetition": host.new_tensor(0.0),
        }
    else:
        decoded = _decoded_structure_penalties(
            outputs,
            x_t,
            batch,
            rhythm_vocab,
            pitch_codec,
            subsample_notes=max(0, int(structure_loss_subsample_notes)),
            subsample_pairs=max(0, int(structure_loss_subsample_pairs)),
        )
    total = host + harm + decoded["duplicate"] + decoded["voice_leading"] + decoded["repetition"]
    return {
        "total": total,
        "host": host,
        "harmonic": harm,
        "duplicate": decoded["duplicate"],
        "voice_leading": decoded["voice_leading"],
        "repetition": decoded["repetition"],
        "full_decoded_executed": 0.0 if fast_music_loss_only else 1.0,
    }
