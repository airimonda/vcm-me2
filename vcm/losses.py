import torch
import torch.nn.functional as F

from .labels import OOS_IDX


def multitask_loss(cmd_logits, slot_logits, cmd_t, head_t, slot_t, label_smoothing: float = 0.1,
                   misfire_weight: float = 0.0, oos_idx: int = OOS_IDX):
    """CE(command) + masked CE(slot) + misfire_weight * misfire term.

    cmd_logits (B,20), slot_logits (B,6,3); cmd_t, head_t, slot_t (B,) int64.
    The slot term uses the head of the TRUE command and only rows with a valid slot
    value (head >= 0 and slot >= 0); other rows (non-slotted commands, OOS, "other slot
    value" rows) contribute nothing. Mean over valid rows (0 if there are none).

    Misfire term (off when misfire_weight == 0): for rows whose TRUE command is OUT_OF_SCOPE,
    mean over those rows of (1 - softmax(cmd_logits)[OOS]) = probability mass on any command.
    Computed from the float logits with no label smoothing; 0 if the batch has no OOS rows.

    Returns (total, l_cmd, l_slot, l_misfire). l_misfire is the term as added to the total,
    i.e. already multiplied by misfire_weight, so total == l_cmd + l_slot + l_misfire.
    """
    cmd_logits = cmd_logits.float()
    l_cmd = F.cross_entropy(cmd_logits, cmd_t, label_smoothing=label_smoothing)
    valid = (head_t >= 0) & (slot_t >= 0)
    if bool(valid.any()):
        logits = slot_logits[valid, head_t[valid]]
        l_slot = F.cross_entropy(logits, slot_t[valid], label_smoothing=label_smoothing)
    else:
        l_slot = slot_logits.sum() * 0.0
    oos = cmd_t == oos_idx
    if misfire_weight != 0.0 and bool(oos.any()):
        p_oos = torch.softmax(cmd_logits[oos], dim=-1)[:, oos_idx]
        l_mis = misfire_weight * (1.0 - p_oos).mean()
    else:
        l_mis = cmd_logits.new_zeros(())
    return l_cmd + l_slot + l_mis, l_cmd, l_slot, l_mis
