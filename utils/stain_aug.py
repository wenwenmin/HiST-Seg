from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is available in the project env, keep fallback safe.
    cv2 = None


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def hed_jitter(image_rgb: np.ndarray, sigma: float = 0.02, bias: float = 0.02, p: float = 0.7) -> np.ndarray:
    """Stain-space augmentation with a safe RGB fallback.

    Parameters are deliberately generic because this function is also used by older
    configs. ``strong_color_augment`` below wraps it with a stronger, configurable
    histology augmentation policy for cross-domain training.
    """
    image = _clip01(image_rgb)
    if np.random.rand() >= float(p):
        return image
    try:
        from skimage.color import hed2rgb, rgb2hed

        hed = rgb2hed(image)
        scale = 1.0 + np.random.normal(0.0, float(sigma), size=(1, 1, 3)).astype(np.float32)
        shift = np.random.normal(0.0, float(bias), size=(1, 1, 3)).astype(np.float32)
        return _clip01(hed2rgb(hed * scale + shift))
    except Exception:
        scale = 1.0 + np.random.normal(0.0, float(sigma), size=(1, 1, 3)).astype(np.float32)
        shift = np.random.normal(0.0, float(bias), size=(1, 1, 3)).astype(np.float32)
        return _clip01(image * scale + shift)


def _brightness_contrast(image: np.ndarray, brightness: float, contrast: float, p: float) -> np.ndarray:
    if np.random.rand() >= float(p):
        return image
    b = 1.0 + np.random.uniform(-float(brightness), float(brightness))
    c = 1.0 + np.random.uniform(-float(contrast), float(contrast))
    mean = image.mean(axis=(0, 1), keepdims=True)
    return _clip01((image - mean) * c + mean * b)


def _hsv_jitter(image: np.ndarray, saturation: float, hue: float, p: float) -> np.ndarray:
    if cv2 is None or np.random.rand() >= float(p):
        return image
    hsv = cv2.cvtColor(_clip01(image), cv2.COLOR_RGB2HSV).astype(np.float32)
    # OpenCV HSV float: H in [0, 360], S/V in [0, 1].
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-float(hue), float(hue)) * 360.0) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + np.random.uniform(-float(saturation), float(saturation))), 0.0, 1.0)
    return _clip01(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


def _gamma_jitter(image: np.ndarray, gamma: float, p: float) -> np.ndarray:
    if np.random.rand() >= float(p):
        return image
    # gamma controls log-space perturbation, e.g. 0.20 -> approx [0.82, 1.22].
    g = float(np.exp(np.random.uniform(-float(gamma), float(gamma))))
    return _clip01(np.power(np.maximum(image, 1e-6), g))


def _blur_jitter(image: np.ndarray, p: float) -> np.ndarray:
    if cv2 is None or np.random.rand() >= float(p):
        return image
    # Mild blur only; the task is boundary-sensitive, so avoid aggressive smoothing.
    sigma = float(np.random.uniform(0.15, 0.65))
    return _clip01(cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma))


def _noise_jitter(image: np.ndarray, std: float, p: float) -> np.ndarray:
    if np.random.rand() >= float(p):
        return image
    noise = np.random.normal(0.0, float(std), size=image.shape).astype(np.float32)
    return _clip01(image + noise)


def strong_color_augment(
    image_rgb: np.ndarray,
    *,
    hed_p: float = 0.90,
    hed_sigma: float = 0.06,
    hed_bias: float = 0.06,
    bc_p: float = 0.85,
    brightness: float = 0.22,
    contrast: float = 0.22,
    hsv_p: float = 0.55,
    saturation: float = 0.18,
    hue: float = 0.035,
    gamma_p: float = 0.45,
    gamma: float = 0.20,
    blur_p: float = 0.15,
    noise_p: float = 0.15,
    noise_std: float = 0.012,
) -> np.ndarray:
    """Stronger histology color augmentation for cross-dataset robustness.

    The policy intentionally combines stain-space and RGB/HSV perturbations. It is
    still conservative on blur/noise because pixel-level tumor boundaries are the
    downstream target. Input and output are RGB float arrays in [0, 1].
    """
    image = _clip01(image_rgb)
    image = hed_jitter(image, sigma=hed_sigma, bias=hed_bias, p=hed_p)
    # Randomize operation order for extra color-domain diversity.
    ops = [
        lambda x: _brightness_contrast(x, brightness=brightness, contrast=contrast, p=bc_p),
        lambda x: _hsv_jitter(x, saturation=saturation, hue=hue, p=hsv_p),
        lambda x: _gamma_jitter(x, gamma=gamma, p=gamma_p),
    ]
    np.random.shuffle(ops)
    for op in ops:
        image = op(image)
    image = _blur_jitter(image, p=blur_p)
    image = _noise_jitter(image, std=noise_std, p=noise_p)
    return _clip01(image)
