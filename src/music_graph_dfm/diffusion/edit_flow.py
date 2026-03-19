"""Edit-flow state transitions, loss, and CTMC sampling."""

from __future__ import annotations

import math
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


def _sample_different_int(rng: random.Random, upper: int, current: int) -> int:
    upper = max(1, int(upper))
    current = int(current)
    if upper <= 1:
        return current
    candidate = rng.randrange(upper)
    if candidate == current:
        candidate = (candidate + 1) % upper
    return candidate


def random_edit_augmentation_step(state: FSNTGV2State, vocab_sizes: Dict[str, int], rng: random.Random) -> FSNTGV2State:
    """Optional augmentation utility (not the EditFlow forward process)."""
    out = state.copy()
    move_choices: List[EditMove] = []

    if out.num_notes > 0:
        i = rng.randrange(out.num_notes)
        move_choices.append(EditMove(move_type=EditMoveType.DELETE_NOTE, note_idx=i))
        move_choices.append(
            EditMove(
                move_type=EditMoveType.SUBSTITUTE_CONTENT,
                note_idx=i,
                pitch_token=rng.randrange(max(1, vocab_sizes["note.pitch_token"])),
                velocity=rng.randrange(max(1, vocab_sizes["note.velocity"])),
                role=rng.randrange(max(1, vocab_sizes["note.role"])),
            )
        )
        move_choices.append(
            EditMove(
                move_type=EditMoveType.SUBSTITUTE_HOST,
                note_idx=i,
                host=rng.randrange(max(1, vocab_sizes["note.host"])),
            )
        )
        move_choices.append(
            EditMove(
                move_type=EditMoveType.SUBSTITUTE_TEMPLATE,
                note_idx=i,
                template=rng.randrange(max(1, vocab_sizes["note.template"])),
            )
        )

    move_choices.append(
        EditMove(
            move_type=EditMoveType.INSERT_NOTE,
            host=rng.randrange(max(1, vocab_sizes["note.host"])),
            template=rng.randrange(max(1, vocab_sizes["note.template"])),
            pitch_token=rng.randrange(max(1, vocab_sizes["note.pitch_token"])),
            velocity=rng.randrange(max(1, vocab_sizes["note.velocity"])),
            role=rng.randrange(max(1, vocab_sizes["note.role"])),
        )
    )

    if out.num_spans > 0:
        move_choices.append(
            EditMove(
                move_type=EditMoveType.SUBSTITUTE_SPAN_RELATION,
                span_src=rng.randrange(out.num_spans),
                span_dst=rng.randrange(out.num_spans),
                relation=rng.randrange(max(1, vocab_sizes["e_ss.relation"])),
            )
        )

    return apply_edit_move(out, rng.choice(move_choices))


def sample_forward_edit_ctmc_step_from_prior(
    state: FSNTGV2State,
    vocab_sizes: Dict[str, int],
    rng: random.Random,
    h: float = 1.0,
    type_rates: Dict[EditMoveType, float] | None = None,
) -> tuple[FSNTGV2State, EditMove | None]:
    """Sample one edit from a forward edit CTMC prior and apply it."""
    rates = {
        EditMoveType.INSERT_NOTE: 0.25,
        EditMoveType.DELETE_NOTE: 0.2,
        EditMoveType.SUBSTITUTE_CONTENT: 0.25,
        EditMoveType.SUBSTITUTE_HOST: 0.1,
        EditMoveType.SUBSTITUTE_TEMPLATE: 0.1,
        EditMoveType.SUBSTITUTE_SPAN_RELATION: 0.1,
    }
    if type_rates is not None:
        rates.update(type_rates)

    valid: Dict[EditMoveType, float] = {}
    if state.num_spans > 0:
        valid[EditMoveType.INSERT_NOTE] = max(0.0, float(rates[EditMoveType.INSERT_NOTE]))
        valid[EditMoveType.SUBSTITUTE_SPAN_RELATION] = max(0.0, float(rates[EditMoveType.SUBSTITUTE_SPAN_RELATION]))
    if state.num_notes > 0:
        valid[EditMoveType.DELETE_NOTE] = max(0.0, float(rates[EditMoveType.DELETE_NOTE]))
        valid[EditMoveType.SUBSTITUTE_CONTENT] = max(0.0, float(rates[EditMoveType.SUBSTITUTE_CONTENT]))
        valid[EditMoveType.SUBSTITUTE_HOST] = max(0.0, float(rates[EditMoveType.SUBSTITUTE_HOST]))
        valid[EditMoveType.SUBSTITUTE_TEMPLATE] = max(0.0, float(rates[EditMoveType.SUBSTITUTE_TEMPLATE]))

    total_hazard = sum(valid.values())
    if total_hazard <= 0:
        return state.copy(), None

    p_jump = 1.0 - math.exp(-float(h) * total_hazard)
    p_jump = max(0.0, min(1.0, p_jump))
    if rng.random() >= p_jump:
        return state.copy(), None

    r = rng.random() * total_hazard
    acc = 0.0
    move_type = EditMoveType.INSERT_NOTE
    for k, v in valid.items():
        acc += v
        if r <= acc:
            move_type = k
            break

    if move_type == EditMoveType.INSERT_NOTE:
        host = rng.randrange(1, max(2, state.num_spans + 1))
        template = rng.randrange(1, max(2, vocab_sizes["note.template"]))
        move = EditMove(
            move_type=move_type,
            host=host,
            template=template,
            pitch_token=rng.randrange(max(1, vocab_sizes["note.pitch_token"])),
            velocity=rng.randrange(max(1, vocab_sizes["note.velocity"])),
            role=rng.randrange(max(1, vocab_sizes["note.role"])),
        )
        return apply_edit_move(state, move), move

    if move_type == EditMoveType.DELETE_NOTE:
        idx = rng.randrange(state.num_notes)
        move = EditMove(move_type=move_type, note_idx=idx)
        return apply_edit_move(state, move), move

    if move_type == EditMoveType.SUBSTITUTE_CONTENT:
        idx = rng.randrange(state.num_notes)
        move = EditMove(
            move_type=move_type,
            note_idx=idx,
            pitch_token=_sample_different_int(
                rng,
                max(1, vocab_sizes["note.pitch_token"]),
                int(state.note_attrs["pitch_token"][idx]),
            ),
            velocity=_sample_different_int(
                rng,
                max(1, vocab_sizes["note.velocity"]),
                int(state.note_attrs["velocity"][idx]),
            ),
            role=_sample_different_int(
                rng,
                max(1, vocab_sizes["note.role"]),
                int(state.note_attrs["role"][idx]),
            ),
        )
        return apply_edit_move(state, move), move

    if move_type == EditMoveType.SUBSTITUTE_HOST:
        idx = rng.randrange(state.num_notes)
        move = EditMove(
            move_type=move_type,
            note_idx=idx,
            host=_sample_different_int(
                rng,
                max(1, min(max(1, vocab_sizes["note.host"]), state.num_spans + 1)),
                int(state.host[idx]),
            ),
        )
        return apply_edit_move(state, move), move

    if move_type == EditMoveType.SUBSTITUTE_TEMPLATE:
        idx = rng.randrange(state.num_notes)
        move = EditMove(
            move_type=move_type,
            note_idx=idx,
            template=_sample_different_int(
                rng,
                max(1, vocab_sizes["note.template"]),
                int(state.template[idx]),
            ),
        )
        return apply_edit_move(state, move), move

    src = rng.randrange(state.num_spans)
    dst = rng.randrange(state.num_spans)
    current_rel = int(state.e_ss[src][dst])
    move = EditMove(
        move_type=EditMoveType.SUBSTITUTE_SPAN_RELATION,
        span_src=src,
        span_dst=dst,
        relation=_sample_different_int(rng, max(1, vocab_sizes["e_ss.relation"]), current_rel),
    )
    return apply_edit_move(state, move), move


def sample_forward_edit_ctmc_trajectory(
    target_state: FSNTGV2State,
    vocab_sizes: Dict[str, int],
    rng: random.Random,
    num_steps: int = 1,
    h: float = 1.0,
    type_rates: Dict[EditMoveType, float] | None = None,
) -> tuple[list[FSNTGV2State], list[EditMove | None]]:
    """Sample a forward edit CTMC trajectory starting at target_state.

    Returns:
        states: [z_0, z_1, ..., z_K] with z_0=target_state and K=num_steps
        moves: forward edits [m_0, ..., m_{K-1}] where z_{k+1}=T(z_k,m_k) (or stay for None)
    """
    out = target_state.copy()
    steps = max(1, int(num_steps))
    states = [out.copy()]
    moves: list[EditMove | None] = []
    for _ in range(steps):
        out, move = sample_forward_edit_ctmc_step_from_prior(
            state=out,
            vocab_sizes=vocab_sizes,
            rng=rng,
            h=h,
            type_rates=type_rates,
        )
        moves.append(move)
        states.append(out.copy())
    return states, moves


def sample_forward_edit_ctmc_source(
    target_state: FSNTGV2State,
    vocab_sizes: Dict[str, int],
    rng: random.Random,
    num_steps: int = 1,
    h: float = 1.0,
    type_rates: Dict[EditMoveType, float] | None = None,
) -> FSNTGV2State:
    """Generate an EditFlow source state by running a forward edit CTMC from target."""
    states, _ = sample_forward_edit_ctmc_trajectory(
        target_state=target_state,
        vocab_sizes=vocab_sizes,
        rng=rng,
        num_steps=num_steps,
        h=h,
        type_rates=type_rates,
    )
    return states[-1]


def sample_multistep_supervision_segment(
    target_state: FSNTGV2State,
    vocab_sizes: Dict[str, int],
    rng: random.Random,
    num_steps: int,
    h: float,
    type_rates: Dict[EditMoveType, float] | None = None,
) -> tuple[FSNTGV2State, FSNTGV2State, EditMove | None, float]:
    """Trajectory-segment supervision for multistep editflow training.

    We sample a forward trajectory z_0 -> ... -> z_K from target z_0.
    A random adjacent segment (z_k, z_{k+1}) is selected and the model is
    supervised to predict one reverse move from z_{k+1} toward z_k.
    """
    states, moves = sample_forward_edit_ctmc_trajectory(
        target_state=target_state,
        vocab_sizes=vocab_sizes,
        rng=rng,
        num_steps=max(2, int(num_steps)),
        h=h,
        type_rates=type_rates,
    )
    valid = [k for k, move in enumerate(moves) if move is not None]
    if not valid:
        k = len(moves) - 1
    else:
        k = valid[rng.randrange(len(valid))]

    source = states[k + 1].copy()
    prev = states[k].copy()
    reverse_move = derive_oracle_edit_move(source, prev)
    t_value = float(k + 1) / float(max(1, len(moves)))
    return source, prev, reverse_move, t_value


def editflow_rate_loss(
    edit_outputs: Dict[str, "torch.Tensor"],
    oracle_moves: Iterable[EditMove | None],
    eps: float = 1e-8,
):
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

    valid_idx = [i for i, m in enumerate(oracle) if m is not None]
    if not valid_idx:
        # No-op batches can happen when forward steps produced only stay transitions.
        return type_logits.sum() * 0.0

    valid_oracle = [oracle[i] for i in valid_idx]
    valid_index_tensor = torch.tensor(valid_idx, dtype=torch.long, device=type_logits.device)
    lam_type = lam_type[valid_index_tensor]
    type_logits = type_logits[valid_index_tensor]

    type_targets = torch.tensor([int(m.move_type) for m in valid_oracle], device=type_logits.device)
    chosen_rate = lam_type[torch.arange(type_targets.shape[0], device=type_logits.device), type_targets]
    poisson = chosen_rate - torch.log(chosen_rate + eps)
    type_ce = F.cross_entropy(type_logits, type_targets)

    arg_losses = []
    for local_b, move in enumerate(valid_oracle):
        b = int(valid_idx[local_b])
        if move.move_type in {EditMoveType.DELETE_NOTE, EditMoveType.SUBSTITUTE_CONTENT, EditMoveType.SUBSTITUTE_HOST, EditMoveType.SUBSTITUTE_TEMPLATE}:
            arg_losses.append(F.cross_entropy(edit_outputs["note_logits"][b : b + 1], torch.tensor([max(0, move.note_idx)], device=type_logits.device)))

        if move.move_type == EditMoveType.SUBSTITUTE_HOST:
            arg_losses.append(
                F.cross_entropy(
                    edit_outputs["host_logits"][b, max(0, move.note_idx)].unsqueeze(0),
                    torch.tensor([move.host], device=type_logits.device),
                )
            )

        if move.move_type == EditMoveType.SUBSTITUTE_TEMPLATE:
            arg_losses.append(
                F.cross_entropy(
                    edit_outputs["template_logits"][b, max(0, move.note_idx)].unsqueeze(0),
                    torch.tensor([move.template], device=type_logits.device),
                )
            )

        if move.move_type == EditMoveType.SUBSTITUTE_CONTENT:
            arg_losses.append(
                F.cross_entropy(
                    edit_outputs["pitch_logits"][b, max(0, move.note_idx)].unsqueeze(0),
                    torch.tensor([move.pitch_token], device=type_logits.device),
                )
            )
            arg_losses.append(
                F.cross_entropy(
                    edit_outputs["velocity_logits"][b, max(0, move.note_idx)].unsqueeze(0),
                    torch.tensor([move.velocity], device=type_logits.device),
                )
            )
            arg_losses.append(
                F.cross_entropy(
                    edit_outputs["role_logits"][b, max(0, move.note_idx)].unsqueeze(0),
                    torch.tensor([move.role], device=type_logits.device),
                )
            )

        if move.move_type == EditMoveType.SUBSTITUTE_SPAN_RELATION:
            arg_losses.append(F.cross_entropy(edit_outputs["span_src_logits"][b : b + 1], torch.tensor([move.span_src], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["span_dst_logits"][b : b + 1], torch.tensor([move.span_dst], device=type_logits.device)))
            arg_losses.append(
                F.cross_entropy(
                    edit_outputs["span_rel_logits"][b, move.span_src, move.span_dst].unsqueeze(0),
                    torch.tensor([move.relation], device=type_logits.device),
                )
            )

        if move.move_type == EditMoveType.INSERT_NOTE:
            arg_losses.append(F.cross_entropy(edit_outputs["insert_host_logits"][b : b + 1], torch.tensor([move.host], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["insert_template_logits"][b : b + 1], torch.tensor([move.template], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["insert_pitch_logits"][b : b + 1], torch.tensor([move.pitch_token], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["insert_velocity_logits"][b : b + 1], torch.tensor([move.velocity], device=type_logits.device)))
            arg_losses.append(F.cross_entropy(edit_outputs["insert_role_logits"][b : b + 1], torch.tensor([move.role], device=type_logits.device)))

    arg_loss = torch.stack(arg_losses).mean() if arg_losses else torch.tensor(0.0, device=type_logits.device)
    return poisson.mean() + type_ce + arg_loss


def _sample_offdiag_from_logits(logits, current: int) -> int | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_sample_offdiag_from_logits requires torch") from exc

    probs = torch.softmax(logits, dim=-1)
    vocab = probs.shape[-1]
    current = max(0, min(vocab - 1, int(current)))
    probs = probs.clone()
    probs[current] = 0.0
    mass = probs.sum()
    if float(mass.item()) <= 1e-12:
        return None
    probs = probs / mass
    return int(torch.distributions.Categorical(probs=probs).sample().item())


def sample_edit_ctmc_step(state: FSNTGV2State, edit_outputs_single: Dict[str, "torch.Tensor"], h: float) -> FSNTGV2State:
    """Sample one off-diagonal edit move from edit rates and apply it."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sample_edit_ctmc_step requires torch") from exc

    lam_type = torch.nn.functional.softplus(edit_outputs_single["lambda_type"][0]).clone()
    valid = torch.ones_like(lam_type, dtype=torch.bool)
    if state.num_spans <= 0:
        valid[int(EditMoveType.INSERT_NOTE)] = False
        valid[int(EditMoveType.SUBSTITUTE_SPAN_RELATION)] = False
    if state.num_notes <= 0:
        valid[int(EditMoveType.DELETE_NOTE)] = False
        valid[int(EditMoveType.SUBSTITUTE_CONTENT)] = False
        valid[int(EditMoveType.SUBSTITUTE_HOST)] = False
        valid[int(EditMoveType.SUBSTITUTE_TEMPLATE)] = False

    lam_type = torch.where(valid, lam_type, torch.zeros_like(lam_type))
    total_hazard = lam_type.sum()
    if float(total_hazard.item()) <= 1e-12:
        return state

    p_jump = 1.0 - torch.exp(-float(h) * total_hazard)
    p_jump = torch.clamp(p_jump, min=0.0, max=1.0)
    if float(torch.rand(1).item()) >= float(p_jump.item()):
        return state

    probs = lam_type / total_hazard.clamp(min=1e-8)
    move_type = int(torch.distributions.Categorical(probs=probs).sample().item())

    if state.num_notes > 0:
        note_logits = edit_outputs_single["note_logits"][0, : state.num_notes]
        note_idx = int(torch.distributions.Categorical(logits=note_logits).sample().item())
    else:
        note_idx = 0

    if move_type == int(EditMoveType.INSERT_NOTE):
        host_logits = edit_outputs_single["insert_host_logits"][0].clone()
        template_logits = edit_outputs_single["insert_template_logits"][0].clone()
        if host_logits.shape[-1] > 1:
            host_logits[0] = -1e9
        if template_logits.shape[-1] > 1:
            template_logits[0] = -1e9
        host = int(torch.distributions.Categorical(logits=host_logits).sample().item())
        template = int(torch.distributions.Categorical(logits=template_logits).sample().item())
        pitch = int(torch.distributions.Categorical(logits=edit_outputs_single["insert_pitch_logits"][0]).sample().item())
        velocity = int(torch.distributions.Categorical(logits=edit_outputs_single["insert_velocity_logits"][0]).sample().item())
        role = int(torch.distributions.Categorical(logits=edit_outputs_single["insert_role_logits"][0]).sample().item())
        if host <= 0 or template <= 0:
            return state
        move = EditMove(EditMoveType.INSERT_NOTE, host=host, template=template, pitch_token=pitch, velocity=velocity, role=role)
        return apply_edit_move(state, move)

    if move_type == int(EditMoveType.DELETE_NOTE):
        if state.num_notes <= 0:
            return state
        return apply_edit_move(state, EditMove(EditMoveType.DELETE_NOTE, note_idx=note_idx))

    if move_type == int(EditMoveType.SUBSTITUTE_CONTENT):
        if state.num_notes <= 0:
            return state
        pitch = _sample_offdiag_from_logits(
            edit_outputs_single["pitch_logits"][0, note_idx],
            int(state.note_attrs["pitch_token"][note_idx]),
        )
        velocity = _sample_offdiag_from_logits(
            edit_outputs_single["velocity_logits"][0, note_idx],
            int(state.note_attrs["velocity"][note_idx]),
        )
        role = _sample_offdiag_from_logits(
            edit_outputs_single["role_logits"][0, note_idx],
            int(state.note_attrs["role"][note_idx]),
        )
        if pitch is None or velocity is None or role is None:
            return state
        move = EditMove(
            EditMoveType.SUBSTITUTE_CONTENT,
            note_idx=note_idx,
            pitch_token=pitch,
            velocity=velocity,
            role=role,
        )
        return apply_edit_move(state, move)

    if move_type == int(EditMoveType.SUBSTITUTE_HOST):
        if state.num_notes <= 0:
            return state
        host = _sample_offdiag_from_logits(edit_outputs_single["host_logits"][0, note_idx], int(state.host[note_idx]))
        if host is None:
            return state
        return apply_edit_move(state, EditMove(EditMoveType.SUBSTITUTE_HOST, note_idx=note_idx, host=host))

    if move_type == int(EditMoveType.SUBSTITUTE_TEMPLATE):
        if state.num_notes <= 0:
            return state
        template = _sample_offdiag_from_logits(
            edit_outputs_single["template_logits"][0, note_idx],
            int(state.template[note_idx]),
        )
        if template is None:
            return state
        return apply_edit_move(state, EditMove(EditMoveType.SUBSTITUTE_TEMPLATE, note_idx=note_idx, template=template))

    if state.num_spans <= 0:
        return state
    src = int(torch.distributions.Categorical(logits=edit_outputs_single["span_src_logits"][0, : state.num_spans]).sample().item())
    dst = int(torch.distributions.Categorical(logits=edit_outputs_single["span_dst_logits"][0, : state.num_spans]).sample().item())
    rel = _sample_offdiag_from_logits(
        edit_outputs_single["span_rel_logits"][0, src, dst],
        int(state.e_ss[src][dst]),
    )
    if rel is None:
        return state
    move = EditMove(EditMoveType.SUBSTITUTE_SPAN_RELATION, span_src=src, span_dst=dst, relation=rel)
    return apply_edit_move(state, move)
