# Robust IDL -> Python port for rotational/macroturbulent/instrumental broadening
import numpy as np
from math import erf

C_KMS = 299792.5
SQRT_PI = np.sqrt(np.pi)
LN2 = np.log(2.0)
FWHM_TO_SIG = 1.0 / (2.0 * np.sqrt(LN2))   # sigma = FWHM / (2*sqrt(ln2))

def _is_equidistant(x, tol_factor=1e-6, max_resol=500000.0):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return True
    dx = np.diff(x)
    dx_min, dx_max = np.min(dx), np.max(dx)
    meanw = 0.5 * (x[0] + x[-1])
    eps = abs(meanw) / max_resol
    return (dx_max - dx_min) <= max(eps, tol_factor * abs(np.median(dx)))

def _make_uniform_grid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if _is_equidistant(x):
        return x.copy(), y.copy(), float(np.median(np.diff(x)))
    step = float(np.median(np.diff(x)))
    n = int(np.round((x[-1] - x[0]) / step)) + 1
    xx = x[0] + step * np.arange(n, dtype=float)
    yy = np.interp(xx, x, y)
    return xx, yy, step

def _linear_convolve_same(y, k):
    """Linear convolution via FFT, centre-aligned 'same' length.
    Pads y and k to common length L = n + m - 1.
    If kernel area ~ 0 (unresolvable), returns y unchanged.
    """
    y = np.asarray(y, dtype=float)
    k = np.asarray(k, dtype=float)
    if y.ndim != 1 or k.ndim != 1:
        raise ValueError("y and k must be 1D arrays")
    s = k.sum()
    if (not np.isfinite(s)) or (abs(s) < 1e-14):
        return y.copy()
    k = k / s
    n, m = y.size, k.size
    L = n + m - 1
    ypad = np.pad(y, (0, L - n), mode='constant')
    kpad = np.pad(k, (0, L - m), mode='constant')
    Y = np.fft.rfft(ypad)
    K = np.fft.rfft(kpad)
    full = np.fft.irfft(Y * K, n=L)
    start = (m - 1) // 2
    end = start + n
    out = full[start:end]
    if out.size != n:
        out = np.convolve(y, k, mode='same')
    return out

def _rotational_kernel(px, lambda0, vrot, beta=1.5):
    if vrot is None or vrot < 0.0:
        return None, 0.0
    dlam_max = vrot * (lambda0 / C_KMS)
    if dlam_max <= 0:
        return None, dlam_max
    x = px / dlam_max
    # Only non-zero for |x|<1
    mask = np.abs(x) < 1.0
    if not np.any(mask):
        return np.zeros_like(px), dlam_max
    x2 = x[mask] * x[mask]
    num = (2.0 * np.sqrt(1.0 - x2) / np.pi) + ((1.0 - x2) * beta / 2.0)
    den = (1.0 + 2.0 * beta / 3.0)
    prof = np.zeros_like(px, dtype=float)
    prof[mask] = (num / den) / dlam_max
    return prof, dlam_max

def _rt_macroturbulence_kernel(px, lambda0, vmacro):
    if vmacro is None or vmacro <= 0:
        return None, 0.0
    MR = vmacro * (lambda0 / C_KMS)
    if MR <= 0:
        return None, MR
    pxmr = np.abs(px) / MR
    prof = (2.0 / (SQRT_PI * MR)) * (np.exp(-pxmr**2) + SQRT_PI * pxmr * (np.vectorize(erf)(pxmr) - 1.0))
    prof = np.maximum(prof, 0.0)
    return prof, MR

def _gaussian_kernel(px, lambda0, vdop=None, ain=None, resol=None):
    if (vdop is None) and (ain is None) and (resol is None):
        return None, 0.0
    if ain is not None:
        dlam_sigma = float(ain) * FWHM_TO_SIG
    elif resol is not None:
        # sigma_v = c / (R * 2*sqrt(ln2))
        sigma_v = C_KMS * FWHM_TO_SIG / float(resol)
        dlam_sigma = sigma_v * (lambda0 / C_KMS)
    else:
        dlam_sigma = float(vdop) * (lambda0 / C_KMS)
    if dlam_sigma <= 0:
        return None, dlam_sigma
    xx = px / dlam_sigma
    prof = np.exp(-(xx**2)) / (np.sqrt(np.pi) * dlam_sigma)
    # truncate far wings
    prof[np.abs(xx) >= 6.0] = 0.0
    return prof, dlam_sigma

def convol_ber(sx, sy, resol=None, vsini=None, beta=1.5, vmacro=None,
               vdop=None, ain=None, message=False, original=False):
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    order = np.argsort(sx)
    sx, sy = sx[order], sy[order]
    sxx, syy, rdst = _make_uniform_grid(sx, sy)
    nn = sxx.size
    lambda0 = 0.5 * (sxx[0] + sxx[-1])
    px = (np.arange(nn) - (nn // 2)) * rdst

    # Threshold: skip kernels narrower than ~0.3 pixel sigma / half-width
    # This avoids zero-sum discrete kernels.
    MIN_WIDTH = 0.3 * rdst

    # 1) Rotational
    if vsini is not None and vsini > 0:
        rot_k, dlam_max = _rotational_kernel(px, lambda0, vsini, beta=beta)
        if (rot_k is not None) and (dlam_max >= MIN_WIDTH) and np.any(rot_k):
            syy = _linear_convolve_same(syy, rot_k)

    # 2) Macroturbulence
    if vmacro is not None and vmacro > 0:
        macro_k, MR = _rt_macroturbulence_kernel(px, lambda0, vmacro)
        if (macro_k is not None) and (MR >= MIN_WIDTH) and np.any(macro_k):
            syy = _linear_convolve_same(syy, macro_k)

    # 3) Gaussian / Instrumental
    gauss_k, dlam_sigma = _gaussian_kernel(px, lambda0, vdop=vdop, ain=ain, resol=resol)
    if (gauss_k is not None) and (dlam_sigma >= MIN_WIDTH) and np.any(gauss_k):
        syy = _linear_convolve_same(syy, gauss_k)

    if original:
        cy = np.interp(sx, sxx, syy)
        cx = sx
    else:
        cx, cy = sxx, syy
    return cx, cy

def convol_by_steps(wavelength_input, flux_input, resol=None, vsini=None, vmacro=None,
                    step=100.0, pad=10.0, beta=1.5, original=False):
    x = np.asarray(wavelength_input, dtype=float)
    y = np.asarray(flux_input, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    n_chunks = max(1, int(np.ceil(abs(x_max - x_min) / float(step))))

    x_out, y_out = [], []
    for j in range(n_chunks):
        left = x_min + step * j - pad
        right = x_min + step * (j + 1) + pad
        sel = (x >= left) & (x < right)
        if not np.any(sel):
            continue
        cx, cy = convol_ber(x[sel], y[sel], resol=resol, vsini=vsini, beta=beta, vmacro=vmacro, original=original)
        keep_left = x_min + step * j
        keep_right = x_min + step * (j + 1)
        keep = (cx >= keep_left) & (cx < keep_right)
        if not np.any(keep):
            continue
        x_out.append(cx[keep])
        y_out.append(cy[keep])

    if not x_out:
        cx, cy = convol_ber(x, y, resol=resol, vsini=vsini, beta=beta, vmacro=vmacro, original=original)
        return cx, cy

    return np.concatenate(x_out), np.concatenate(y_out)
