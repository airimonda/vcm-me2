import numpy as np
import onnxruntime as ort
import pytest
import torch

from vcm.models import ARCHS, TIER_TARGETS, TIER_TOLERANCE, build_model, check_tier, count_params, BCResNet


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("tier", ["S", "M"])
def test_forward_and_param_count(arch, tier):
    m = build_model(arch, tier).eval()
    n = count_params(m)
    t = TIER_TARGETS[tier]
    print(f"{arch} {tier}: {n}")
    assert abs(n - t) <= TIER_TOLERANCE * t, (arch, tier, n)
    assert check_tier(m, tier)
    x = torch.randn(3, 80000) * 0.1
    cmd, slots = m(x)
    assert cmd.shape == (3, 20) and slots.shape == (3, 6, 3)
    assert torch.isfinite(cmd).all() and torch.isfinite(slots).all()
    # accepts (B,1,64,T) and (B,64,T) log-mel
    f = m.features(x)
    assert f.shape == (3, 64, 498)
    a = m.forward_features(f)
    b = m.forward_features(f.unsqueeze(1))
    assert torch.allclose(a[0], b[0], atol=1e-5)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_backward_runs(arch):
    m = build_model(arch, "S").train()
    f = m.features(torch.randn(4, 80000) * 0.1)
    c, s = m.forward_features(f)
    (c.sum() + s.sum()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_onnx_export_and_parity(arch, tmp_path):
    from vcm.export import export_onnx
    m = build_model(arch, "S").eval()
    p = str(tmp_path / "m.onnx")
    side = export_onnx(m, p, tau=0.4)
    assert side["parity_max_abs_diff"] < 1e-3
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 80000).astype(np.float32) * 0.1
    c, sl = s.run(None, {"waveform": x})
    assert c.shape == (1, 20) and sl.shape == (1, 6, 3)
    assert (tmp_path / "m.json").exists()


def test_bc_resnet_subspectral_norm():
    m = BCResNet(scale=8, subspectral=True)
    from vcm.models import AttnPool
    y = m(torch.randn(2, 64, 498))
    assert y.dim() == 3 and y.shape[1] == m.out_dim


def test_no_pretrained_weights_anywhere():
    # all models are built from random init; nothing is downloaded
    import vcm.models as M
    src = open(M.__file__).read()
    assert "pretrained" not in src.replace("no pretrained", "")
