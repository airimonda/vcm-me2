import torch
import torch.nn.functional as F

from vcm.models import EnsembleModel, build_model


def test_ensemble_of_one_matches_member_probs():
    torch.manual_seed(0)
    m = build_model("tc_resnet", "S").eval()
    e = EnsembleModel([m]).eval()
    x = torch.randn(2, 80000) * 0.1
    with torch.no_grad():
        c, s = m(x)
        ce, se = e(x)
    assert torch.allclose(F.softmax(c, -1), F.softmax(ce, -1), atol=1e-5)
    assert torch.allclose(F.softmax(s, -1), F.softmax(se, -1), atol=1e-5)


def test_ensemble_averages_probabilities():
    torch.manual_seed(0)
    a, b = build_model("tc_resnet", "S").eval(), build_model("tc_resnet", "S").eval()
    x = torch.randn(3, 80000) * 0.1
    with torch.no_grad():
        ce, _ = EnsembleModel([a, b])(x)
        want = (F.softmax(a(x)[0], -1) + F.softmax(b(x)[0], -1)) / 2
    assert torch.allclose(F.softmax(ce, -1), want, atol=1e-5)
