# SPDX-License-Identifier: Apache-2.0
"""The token width of a verification batch with ragged drafts (n-gram lookup, the lookup + MTP hybrid).

Pure helpers, importable on CPU.

A verification batch runs every decode request as one padded row of ``T`` tokens, ``T`` the widest
scheduled row; the recurrent layers (GDN, PLE short conv) read each request's state from column
``num_accepted - 1`` of this step's state lattice, ``num_accepted`` being the previous step's
accepted count, and the kernels clamp that column to ``T - 1`` (``gated_delta_rule_mtp_packed``,
``_gated_delta_rule_mtp_prepared``). An MTP batch is always ``1 + k`` wide, so the clamp never binds.
Ragged drafts can make a step narrower than the previous step's accepted count: a step that accepted
all ``k`` drafts followed by a step whose drafts are shorter (or empty, the ``[-1]`` placeholder)
would load the state of an earlier token. ``verification_width`` widens such a step to the largest
accepted count of its requests; the extra columns are padding rows (past each request's real length:
no state store, dummy KV slots, the QSA scratch ring row).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

__all__ = ["verification_width", "pad_query_rows"]


def verification_width(num_scheduled: Sequence[int], accepted: Sequence[int], recurrent_spec: bool) -> int:
    """Token width of a decode batch: the widest scheduled row; for a verification batch of a model with
    recurrent state (``recurrent_spec`` and a row wider than one token) at least the largest previous
    accepted count, so every request's load column ``accepted - 1`` lies inside the lattice.

    ValueError: no scheduled row, a scheduled row below one token or an accepted count below one."""
    if len(num_scheduled) == 0:
        raise ValueError("a decode batch has at least one request")
    widest = int(max(num_scheduled))
    if min(int(n) for n in num_scheduled) < 1:
        raise ValueError(f"every decode request schedules at least one token, got {list(num_scheduled)}")
    if not recurrent_spec or widest == 1:
        return widest
    if accepted and min(int(a) for a in accepted) < 1:
        raise ValueError(f"accepted counts are at least 1, got {list(accepted)}")
    return max([widest] + [int(a) for a in accepted])


def pad_query_rows(rows: Sequence[torch.Tensor], width: int) -> torch.Tensor:
    """``[len(rows), width]``: each 1-D row right-padded with zeros (``pad_sequence`` pads only to the
    longest row). ValueError: a row longer than ``width``."""
    out = pad_sequence(list(rows), batch_first=True, padding_value=0)
    if out.shape[1] > width:
        raise ValueError(f"a query row has {out.shape[1]} tokens, more than the batch width {width}")
    if out.shape[1] < width:
        out = torch.nn.functional.pad(out, (0, width - out.shape[1]))
    return out
