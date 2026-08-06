"""
desisim.torch_utils
====================

Small, isolated helpers for optional PyTorch/CUDA acceleration.

PyTorch is a *soft* dependency of desisim (see requirements.txt): nothing at
package-import time imports this module or torch itself, so environments
without torch installed are completely unaffected. Only individual
functions that explicitly opt into torch-accelerated code paths (currently
EMSpectrum.spectrum()'s line-profile builder; see templates.py) import it,
and only when actually asked to use the torch backend.

Rationale for keeping this isolated in its own module rather than scattering
torch imports throughout templates.py etc.: it gives a single place to unit
test the device-resolution logic, and a single place to update if/when the
project's device policy changes (e.g. multi-GPU, MPS support).
"""


def torch_available():
    """Return True if PyTorch can be imported in the current environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def get_device(device=None):
    """Resolve a `torch.device` to compute on.

    Parameters
    ----------
    device : str, torch.device, or None
        Explicit device request (e.g. 'cpu', 'cuda', 'cuda:1'). If None
        (default), auto-detects: CUDA if `torch.cuda.is_available()`,
        otherwise CPU. This is the sole place device auto-detection logic
        lives, so it stays consistent across every torch-accelerated code
        path in the project.

    Returns
    -------
    torch.device
    """
    import torch

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
