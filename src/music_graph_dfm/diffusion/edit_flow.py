"""Edit-flow state transitions, loss, and CTMC sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List

from music_graph_dfm.representation.state import FSNTGV2State


class EditMoveType(IntEnum):
    INSERT_NOTE = 0
    DELETE_NOTE = 1
    SUBSTITUTE_CONTENT = 2
    SUBSTITUTE_HOST = 3
    SUBSTITUTE_TEMPLATE = 4
    SUBSTITUTE_SPAN_RELATION = 5


@dataclass(frozen=True)
class EditMove:
    move_type: EditMoveType
    note_idx: int = -1
    host: int = 0
    template: int = 0
    pitch_token: int = 0
    velocity: int = 0
    role: int = 0
    span_src: int = -1
    span_dst: int = -1
    relation: int = 0


def apply_edit_move(state: FSNTGV2State, move: EditMove) -> FSNTGV2State:
    out = state.copy()

    if move.move_type == EditMoveType.INSERT_NOTE:
        out.note_attrs["active"].append(1)
        out.note_attrs["pitch_token"].append(int(move.pitch_token))
        out.note_attrs["velocity"].append(int(move.velocity))
        out.note_attrs["role"].append(int(move.role))
        out.host.append(int(move.host))
        out.template.append(int(move.template))

    elif move.move_type == EditMoveType.DELETE_NOTE:
        if 0 <= move.note_idx < out.num_notes:
            for channel in out.note_attrs:
                out.note_attrs[channel].pop(move.note_idx)
            out.host.pop(move.note_idx)
            out.template.pop(move.note_idx)

    elif move.move_type == EditMoveType.SUBSTITUTE_CONTENT:
        if 0 <= move.note_idx < out.num_notes:
            out.note_attrs["pitch_token"][move.note_idx] = int(move.pitch_token)
            out.note_attrs["velocity"][move.note_idx] = int(move.velocity)
            out.note_attrs["role"][move.note_idx] = int(move.role)

    elif move.move_type == EditMoveType.SUBSTITUTE_HOST:
        if 0 <= move.note_idx < out.num_notes:
            out.host[move.note_idx] = int(move.host)

    elif move.move_type == EditMoveType.SUBSTITUTE_TEMPLATE:
        if 0 <= move.note_idx < out.num_notes:
            out.template[move.note_idx] = int(move.template)

    elif move.move_type == EditMoveType.SUBSTITUTE_SPAN_RELATION:
        if 0 <= move.span_src < out.num_spans and 0 <= move.span_dst < out.num_spans:
            out.e_ss[move.span_src][move.span_dst] = int(move.relation)

    out.validate_shapes()
    out.project_placement_consistency()
    return out


def derive_oracle_edit_move(source: FSNTGV2State, target: FSNTGV2State) -> EditMove | None:
    """Returns one valid edit that moves source toward target."""
    ns = source.num_notes
    nt = target.num_notes

    if ns < nt:
        idx = ns
        return EditMove(
            move_type=EditMoveType.INSERT_NOTE,
            host=int(target.host[idx]),
            template=int(target.template[idx]),
            pitch_token=int(target.note_attrs["pitch_token"][idx]),
            velocity=int(target.note_attrs["velocity"][idx]),
            role=int(target.note_attrs["role"][idx]),
        )

    if ns > nt:
        return EditMove(move_type=EditMoveType.DELETE_NOTE, note_idx=nt)

    for i in range(ns):
        if int(source.note_attrs["active"][i]) != int(target.note_attrs["active"][i]):
            if int(target.note_attrs["active"][i]) == 0:
                return EditMove(move_type=EditMoveType.DELETE_NOTE, note_idx=i)
            return EditMove(
                move_type=EditMoveType.SUBSTITUTE_CONTENT,
                note_idx=i,
                pitch_token=int(target.note_attrs["pitch_token"][i]),
                velocity=int(target.note_attrs["velocity"][i]),
                role=int(target.note_attrs["role"][i]),
            )

        if int(source.host[i]) != int(target.host[i]):
            return EditMove(move_type=EditMoveType.SUBSTITUTE_HOST, note_idx=i, host=int(target.host[i]))

        if int(source.template[i]) != int(target.template[i]):
            return EditMove(move_type=EditMoveType.SUBSTITUTE_TEMPLATE, note_idx=i, template=int(target.template[i]))

        if (
            int(source.note_attrs["pitch_token"][i]) != int(target.note_attrs["pitch_token"][i])
            or int(source.note_attrs["velocity"][i]) != int(target.note_attrs["velocity"][i])
            or int(source.note_attrs["role"][i]) != int(target.note_attrs["role"][i])
        ):
            return EditMove(
                move_type=EditMoveType.SUBSTITUTE_CONTENT,
                note_idx=i,
                pitch_token=int(target.note_attrs["pitch_token"][i]),
                velocity=int(target.note_attrs["velocity"][i]),
                role=int(target.note_attrs["role"][i]),
            )

    n = min(source.num_spans, target.num_spans)
    for i in range(n):
        for j in range(n):
            if int(source.e_ss[i][j]) != int(target.e_ss[i][j]):
                return EditMove(
                    move_type=EditMoveType.SUBSTITUTE_SPAN_RELATION,
                    span_src=i,
                    span_dst=j,
                    relation=int(target.e_ss[i][j]),
                )

    return None


def perturb_state_for_editflow(state: FSNTGV2State, vocab_sizes: Dict[str, int], rng: random.Random) -> FSNTGV2State:
    """Samples a neighboring edit state as EditFlow source state."""
    out = state.copy()
    move_choices: List[EditMove] = []

    if out.num_notes > 0:
        i = rng.randrange(out.num_notes)
        move_choices.append(EditMove(move_type=EditMoveType.DELETE_NOTE, note_idx=i))
        move_choices.append(EditMove(
            move_type=EditMoveType.SUBSTITUTE_CONTENT,
            note_idx=i,
            pitch_token=rng.randrange(max(1, vocab_sizes["note.pitch_token"])),
            velocity=rng.randrange(max(1, vocab_sizes["note.velocity"])),
            role=rng.randrange(max(1, vocab_sizes["note.role"])),
        ))
        move_choices.append(EditMove(
            move_type=EditMoveType.SUBSTITUTE_HOST,
            note_idx=i,
            host=rng.randrange(max(1, vocab_sizes["note.host"])),
        ))
        move_choices.append(EditMove(
            move_type=EditMoveType.SUBSTITUTE_TEMPLATE,
            note_idx=i,
            template=rng.randrange(max(1, vocab_sizes["note.template"])),
        ))

    move_choices.append(EditMove(
        move_type=EditMoveType.INSERT_NOTE,
        host=rng.randrange(max(1, vocab_sizes["note.host"])),
        template=rng.randrange(max(1, vocab_sizes["note.template"])),
        pitch_token=rng.randrange(max(1, vocab_sizes["note.pitch_token"])),
        velocity=rng.randrange(max(1, vocab_sizes["note.velocity"])),
        role=rng.randrange(max(1, vocab_sizes["note.role"])),
    ))

    if out.num_spans > 0:
        move_choices.append(EditMove(
            move_type=EditMoveType.SUBSTITUTE_SPAN_RELATION,
            span_src=rng.randrange(out.num_spans),
            span_dst=rng.randrange(out.num_spans),
            relation=rng.randrange(max(1, vocab_sizes["e_ss.relation"])),
        ))

    move = rng.choice(move_choices)
    return apply_edit_move(out, move)


def editflow_rate_loss(edit_outputs: Dict[str, "torch.Tensor"], oracle_moves: Iterable[EditMove], eps: float = 1e-8):
    """Edit-flow objective with type-level Poisson rate loss + argument CE losses."""
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("editflow_rate_loss requires torch") from exc

    oracle = list(oracle_moves)
    batch_size = len(oracle)
    if batch_size == 0:
        return torch.tensor(0.0)

    lam_type = torch.nn.functional.softplus(edit_outputs["lambda_type"])  # [B, T]
    type_logits = edit_outputs["type_logits"]

    type_targets = torch.tensor([int(m.move_type) if m is not None else 0 for m in oracle], device=type_logits.device)
    chosen_rate = lam_type[torch.arange(batch_size, device=type_logits.device), type_targets]
    poisson = chosen_rate - torch.log(chosen_rate + eps)
    type_ce = F.cross_entropy(type_logits, type_targets)

    arg_losses = []
    for b, move in enumerate(oracle):
        if move is None:
            continue
        if move.move_type in {EditMoveType.DELETE_NOTE, EditMoveType.SUBSTITUTE_CONTENT, EditMoveType.SUBSTITUTE_HOST, EditMoveType.SUBSTITUTE_TEMPLATE}:
            arg_losses.append(F.cross_entropy(edit_outputs["note_logits"][b : b + 1], torch.tensor([max(0, move.note_idx)], device=type_logits.device)))

        if move.move_type == EditMoveType.SUBSTITUTE_HOST:
            arg_losses.append(F.cross_entropy(edit_outputs["host_logits"][b, max(0, move.note_idx)].unsqueeze(0), torch.tensor([move.host], device=type_logits.device)))

        if move.move_type == EditMoveType.SUBSTITUTE_TEMPLATE:
            arg_losses.append(F.cross_entropy(edit_outputs["template_logits"][b, max(0, move.note_idx)].unsqueeze(0), torch.tensor([move.template], device=type_logits.device)))

        if move.move_type in {EditMoveType.INSERT_NOTE, EditMoveType.SUBSTITUTE_CONTENT}:
            arg_losses.append(F.cross_entropy(edit_outputs["pitch_logits"][b, max(0, move.note_idx)].unsqueeze(0), torch.tensor([move.pitch_token], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["velocity_logits"][b, max(0, move.note_idx)].unsqueeze(0), torch.tensor([move.velocity], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["role_logits"][b, max(0, move.note_idx)].unsqueeze(0), torch.tensor([move.role], device=type_logits.device)))

        if move.move_type == EditMoveType.SUBSTITUTE_SPAN_RELATION:
            arg_losses.append(F.cross_entropy(edit_outputs["span_src_logits"][b : b + 1], torch.tensor([move.span_src], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["span_dst_logits"][b : b + 1], torch.tensor([move.span_dst], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["span_rel_logits"][b : b + 1], torch.tensor([move.relation], device=type_logits.device)))

        if move.move_type == EditMoveType.INSERT_NOTE:
            arg_losses.append(F.cross_entropy(edit_outputs["insert_host_logits"][b : b + 1], torch.tensor([move.host], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["insert_template_logits"][b : b + 1], torch.tensor([move.template], device=type_logits.device)))

    arg_loss = torch.stack(arg_losses).mean() if arg_losses else torch.tensor(0.0, device=type_logits.device)
    return poisson.mean() + type_ce + arg_loss


def sample_edit_ctmc_step(state: FSNTGV2State, edit_outputs_single: Dict[str, "torch.Tensor"], h: float) -> FSNTGV2State:
    """Sample one off-diagonal edit move from edit rates and apply it."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sample_edit_ctmc_step requires torch") from exc

    lam_type = torch.nn.functional.softplus(edit_outputs_single["lambda_type"][0])
    total_hazard = lam_type.sum()
    p_jump = 1.0 - torch.exp(-float(h) * total_hazard)
    if float(torch.rand(1).item()) >= float(p_jump.item()):
        return state

    probs = lam_type / lam_type.sum().clamp(min=1e-8)
    move_type = int(torch.distributions.Categorical(probs=probs).sample().item())

    note_logits = edit_outputs_single["note_logits"][0]
    note_idx = int(torch.distributions.Categorical(logits=note_logits).sample().item()) if state.num_notes > 0 else 0

    if move_type == int(EditMoveType.INSERT_NOTE):
        host = int(torch.distributions.Categorical(logits=edit_outputs_single["insert_host_logits"][0]).sample().item())
        template = int(torch.distributions.Categorical(logits=edit_outputs_single["insert_template_logits"][0]).sample().item())
        pitch = int(torch.distributions.Categorical(logits=edit_outputs_single["pitch_logits"][0, 0]).sample().item())
        velocity = int(torch.distributions.Categorical(logits=edit_outputs_single["velocity_logits"][0, 0]).sample().item())
        role = int(torch.distributions.Categorical(logits=edit_outputs_single["role_logits"][0, 0]).sample().item())
        move = EditMove(EditMoveType.INSERT_NOTE, host=host, template=template, pitch_token=pitch, velocity=velocity, role=role)
        return apply_edit_move(state, move)

    if move_type == int(EditMoveType.DELETE_NOTE):
        return apply_edit_move(state, EditMove(EditMoveType.DELETE_NOTE, note_idx=note_idx))

    if move_type == int(EditMoveType.SUBSTITUTE_CONTENT):
        pitch_logits = edit_outputs_single["pitch_logits"][0, note_idx].clone()
        velocity_logits = edit_outputs_single["velocity_logits"][0, note_idx].clone()
        role_logits = edit_outputs_single["role_logits"][0, note_idx].clone()

        current_pitch = int(state.note_attrs["pitch_token"][note_idx]) if note_idx < state.num_notes else 0
        current_velocity = int(state.note_attrs["velocity"][note_idx]) if note_idx < state.num_notes else 0
        current_role = int(state.note_attrs["role"][note_idx]) if note_idx < state.num_notes else 0

        if current_pitch < pitch_logits.shape[-1]:
            pitch_logits[current_pitch] = -1e9
        if current_velocity < velocity_logits.shape[-1]:
            velocity_logits[current_velocity] = -1e9
        if current_role < role_logits.shape[-1]:
            role_logits[current_role] = -1e9

        move = EditMove(
            EditMoveType.SUBSTITUTE_CONTENT,
            note_idx=note_idx,
            pitch_token=int(torch.distributions.Categorical(logits=pitch_logits).sample().item()),
            velocity=int(torch.distributions.Categorical(logits=velocity_logits).sample().item()),
            role=int(torch.distributions.Categorical(logits=role_logits).sample().item()),
        )
        return apply_edit_move(state, move)

    if move_type == int(EditMoveType.SUBSTITUTE_HOST):
        logits = edit_outputs_single["host_logits"][0, note_idx].clone()
        current = int(state.host[note_idx]) if note_idx < state.num_notes else 0
        if current < logits.shape[-1]:
            logits[current] = -1e9
        host = int(torch.distributions.Categorical(logits=logits).sample().item())
        return apply_edit_move(state, EditMove(EditMoveType.SUBSTITUTE_HOST, note_idx=note_idx, host=host))

    if move_type == int(EditMoveType.SUBSTITUTE_TEMPLATE):
        logits = edit_outputs_single["template_logits"][0, note_idx].clone()
        current = int(state.template[note_idx]) if note_idx < state.num_notes else 0
        if current < logits.shape[-1]:
            logits[current] = -1e9
        template = int(torch.distributions.Categorical(logits=logits).sample().item())
        return apply_edit_move(state, EditMove(EditMoveType.SUBSTITUTE_TEMPLATE, note_idx=note_idx, template=template))

    src = int(torch.distributions.Categorical(logits=edit_outputs_single["span_src_logits"][0]).sample().item())
    dst = int(torch.distributions.Categorical(logits=edit_outputs_single["span_dst_logits"][0]).sample().item())
    rel_logits = edit_outputs_single["span_rel_logits"][0, src, dst].clone()
    current_rel = int(state.e_ss[src][dst]) if src < state.num_spans and dst < state.num_spans else 0
    if current_rel < rel_logits.shape[-1]:
        rel_logits[current_rel] = -1e9
    rel = int(torch.distributions.Categorical(logits=rel_logits).sample().item())
    move = EditMove(EditMoveType.SUBSTITUTE_SPAN_RELATION, span_src=src, span_dst=dst, relation=rel)
    return apply_edit_move(state, move)
