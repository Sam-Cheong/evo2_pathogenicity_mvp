"""Regression for provider Base64 transport formatting, not real model metrics."""
import base64
import io

import numpy as np
import pytest

from evo2_mvp.nvidia_encoder import decode_base64_npz, NvidiaEvo2Encoder


def test_line_wrapped_base64_remains_strict_and_reloadable():
    stream = io.BytesIO()
    array = np.arange(12, dtype=np.float64).reshape(1, 4, 3)
    np.savez(stream, **{"blocks.20.mlp.l3.output": array})
    text = base64.encodebytes(stream.getvalue()).decode("ascii")
    assert "\n" in text
    with np.load(io.BytesIO(decode_base64_npz(text)), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["blocks.20.mlp.l3.output"], array)
    with pytest.raises(ValueError):
        decode_base64_npz(text[:10] + "!" + text[10:])
    with pytest.raises(ValueError):
        decode_base64_npz("\u200b" + text)

