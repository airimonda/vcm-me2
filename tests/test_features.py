import numpy as np
import onnxruntime as ort
import torch

from vcm.features import LogMel


def test_shape_and_norm():
    m = LogMel()
    x = torch.randn(2, 80000) * 0.1
    y = m(x)
    assert y.shape == (2, 64, 498)
    assert torch.allclose(y.mean(dim=(1, 2)), torch.zeros(2), atol=1e-4)


def test_stft_matches_torch_stft():
    m = LogMel(norm=False)
    x = torch.randn(1, 16000) * 0.1
    ref = torch.stft(x, 400, 160, window=torch.hann_window(400), center=False, return_complex=True).abs() ** 2
    spec = torch.nn.functional.conv1d(x.unsqueeze(1), m.dft, stride=160)
    pw = spec[:, :201] ** 2 + spec[:, 201:] ** 2
    assert torch.allclose(pw, ref, rtol=1e-3, atol=1e-4)


def test_logmel_onnx_parity(tmp_path):
    m = LogMel().eval()
    x = torch.randn(2, 80000) * 0.1
    p = str(tmp_path / "lm.onnx")
    torch.onnx.export(m, (x,), p, opset_version=17, input_names=["wav"], output_names=["feats"],
                      dynamic_axes={"wav": {0: "b"}, "feats": {0: "b"}}, dynamo=False)
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    for batch in (2, 1, 3):
        xb = torch.randn(batch, 80000) * 0.1
        got = s.run(None, {"wav": xb.numpy()})[0]
        assert np.abs(got - m(xb).numpy()).max() < 1e-3
