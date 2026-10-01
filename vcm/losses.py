import torch
import torch.nn.functional as F


def multitask_loss(cmd_logits, slot_logits, cmd_t, head_t, slot_t, label_smoothing: float = 0.1):
    """CE(command) + masked CE(slot).

    cmd_logits (B,20), slot_logits (B,6,3); cmd_t, head_t, slot_t (B,) int64.
    The slot term uses the head of the TRUE command and only rows with a valid slot
    value (head >= 0 and slot >= 0); other rows (non-slotted commands, OOS, "other slot
    value" rows) contribute nothing. Mean over valid rows (0 if there are none).
    """
    l_cmd = F.cross_entropy(cmd_logits, cmd_t, label_smoothing=label_smoothing)
    valid = (head_t >= 0) & (slot_t >= 0)
    if bool(valid.any()):
        logits = slot_logits[valid, head_t[valid]]
        l_slot = F.cross_entropy(logits, slot_t[valid], label_smoothing=label_smoothing)
    else:
        l_slot = slot_logits.sum() * 0.0
    return l_cmd + l_slot, l_cmd, l_slot
