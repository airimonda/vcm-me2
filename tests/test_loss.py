import torch

from vcm.losses import multitask_loss


def test_slot_loss_masked_for_invalid_rows():
    torch.manual_seed(0)
    cmd = torch.randn(4, 20)
    slots = torch.randn(4, 6, 3, requires_grad=True)
    cmd_t = torch.tensor([0, 19, 13, 13])
    head_t = torch.tensor([-1, -1, 0, 0])
    slot_t = torch.tensor([-1, -1, 1, -1])          # last row: slotted command, "other slot value"
    loss, lc, ls, _ = multitask_loss(cmd, slots, cmd_t, head_t, slot_t, 0.0)
    loss.backward()
    g = slots.grad
    assert g[0].abs().sum() == 0 and g[1].abs().sum() == 0 and g[3].abs().sum() == 0
    assert g[2, 0].abs().sum() > 0 and g[2, 1:].abs().sum() == 0      # only the true command's head
    expect = torch.nn.functional.cross_entropy(slots[2:3, 0].detach(), torch.tensor([1]))
    assert torch.allclose(ls, expect)


def test_no_valid_rows_gives_zero_slot_loss():
    cmd = torch.randn(3, 20)
    slots = torch.randn(3, 6, 3, requires_grad=True)
    loss, lc, ls, _ = multitask_loss(cmd, slots, torch.tensor([0, 1, 2]), torch.full((3,), -1), torch.full((3,), -1))
    assert ls.item() == 0.0 and torch.allclose(loss, lc)
    loss.backward()


def test_slot_loss_is_mean_over_valid():
    cmd = torch.zeros(2, 20)
    slots = torch.randn(2, 6, 3)
    _, _, ls, _ = multitask_loss(cmd, slots, torch.tensor([13, 15]), torch.tensor([0, 2]), torch.tensor([0, 2]), 0.0)
    a = torch.nn.functional.cross_entropy(slots[0, 0:1], torch.tensor([0]))
    b = torch.nn.functional.cross_entropy(slots[1, 2:3], torch.tensor([2]))
    assert torch.allclose(ls, (a + b) / 2)


# ------------------------------------------------------------------ misfire term
def _args(cmd_t):
    n = len(cmd_t)
    return torch.tensor(cmd_t), torch.full((n,), -1), torch.full((n,), -1)


def test_misfire_zero_when_weight_zero_or_no_oos_rows():
    torch.manual_seed(1)
    cmd, slots = torch.randn(4, 20), torch.randn(4, 6, 3)
    ct, ht, st = _args([19, 3, 19, 5])
    a = multitask_loss(cmd, slots, ct, ht, st, 0.1)
    b = multitask_loss(cmd, slots, ct, ht, st, 0.1, misfire_weight=0.0)
    assert a[3].item() == 0.0 and b[3].item() == 0.0 and a[0].item() == b[0].item()
    ct, ht, st = _args([0, 3, 7, 5])                                  # no OOS rows
    out = multitask_loss(cmd, slots, ct, ht, st, 0.1, misfire_weight=2.0)
    assert out[3].item() == 0.0 and torch.allclose(out[0], out[1] + out[2])


def test_misfire_matches_hand_computed_value():
    cmd = torch.zeros(3, 20)
    cmd[0, 19] = 2.0                         # OOS row 0
    cmd[1, 4] = 1.5                          # OOS row 1 (leans to a command)
    cmd[2, 19] = 9.0                         # in-scope row: must not enter the term
    ct, ht, st = _args([19, 19, 4])
    slots = torch.zeros(3, 6, 3)
    w = 0.7
    out = multitask_loss(cmd, slots, ct, ht, st, 0.0, misfire_weight=w)
    p0 = torch.softmax(cmd[0], -1)[19].item()
    p1 = torch.softmax(cmd[1], -1)[19].item()
    expect = w * ((1 - p0) + (1 - p1)) / 2
    assert abs(out[3].item() - expect) < 1e-6
    assert torch.allclose(out[0], out[1] + out[2] + out[3])
    base = multitask_loss(cmd, slots, ct, ht, st, 0.0, misfire_weight=0.0)
    assert abs((out[0] - base[0]).item() - expect) < 1e-6          # "train_loss" includes the term


def test_misfire_gradient_pushes_oos_prob_up():
    torch.manual_seed(2)
    cmd = torch.randn(3, 20, requires_grad=True)
    ct, ht, st = _args([19, 2, 19])
    slots = torch.zeros(3, 6, 3)
    multitask_loss(cmd, slots, ct, ht, st, 0.0, misfire_weight=1.0)[3].backward()
    g = cmd.grad
    assert (g[0, 19] < 0).all() and (g[2, 19] < 0).all()             # descending raises the OOS logit
    assert (g[0, :19] >= 0).all() and (g[2, :19] >= 0).all()          # and lowers the command logits
    assert g[1].abs().sum() == 0                                      # in-scope row untouched
    cmd2 = cmd.detach().clone().requires_grad_(True)
    l = multitask_loss(cmd2, slots, ct, ht, st, 0.0, misfire_weight=1.0)[3]
    l.backward()
    with torch.no_grad():
        stepped = cmd2 - 0.5 * cmd2.grad
    assert torch.softmax(stepped, -1)[[0, 2], 19].mean() > torch.softmax(cmd2.detach(), -1)[[0, 2], 19].mean()
