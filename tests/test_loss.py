import torch

from vcm.losses import multitask_loss


def test_slot_loss_masked_for_invalid_rows():
    torch.manual_seed(0)
    cmd = torch.randn(4, 20)
    slots = torch.randn(4, 6, 3, requires_grad=True)
    cmd_t = torch.tensor([0, 19, 13, 13])
    head_t = torch.tensor([-1, -1, 0, 0])
    slot_t = torch.tensor([-1, -1, 1, -1])          # last row: slotted command, "other slot value"
    loss, lc, ls = multitask_loss(cmd, slots, cmd_t, head_t, slot_t, 0.0)
    loss.backward()
    g = slots.grad
    assert g[0].abs().sum() == 0 and g[1].abs().sum() == 0 and g[3].abs().sum() == 0
    assert g[2, 0].abs().sum() > 0 and g[2, 1:].abs().sum() == 0      # only the true command's head
    expect = torch.nn.functional.cross_entropy(slots[2:3, 0].detach(), torch.tensor([1]))
    assert torch.allclose(ls, expect)


def test_no_valid_rows_gives_zero_slot_loss():
    cmd = torch.randn(3, 20)
    slots = torch.randn(3, 6, 3, requires_grad=True)
    loss, lc, ls = multitask_loss(cmd, slots, torch.tensor([0, 1, 2]), torch.full((3,), -1), torch.full((3,), -1))
    assert ls.item() == 0.0 and torch.allclose(loss, lc)
    loss.backward()


def test_slot_loss_is_mean_over_valid():
    cmd = torch.zeros(2, 20)
    slots = torch.randn(2, 6, 3)
    _, _, ls = multitask_loss(cmd, slots, torch.tensor([13, 15]), torch.tensor([0, 2]), torch.tensor([0, 2]), 0.0)
    a = torch.nn.functional.cross_entropy(slots[0, 0:1], torch.tensor([0]))
    b = torch.nn.functional.cross_entropy(slots[1, 2:3], torch.tensor([2]))
    assert torch.allclose(ls, (a + b) / 2)
