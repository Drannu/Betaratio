
import numpy as np
from typing import Tuple, Optional, overload, Literal
import numpy.typing as npt

C_KMS = 299792.458  # km/s

# -----------------------------
# Utility kernels (velocity space)
# -----------------------------

def _gaussian_kernel_velocity(dv_kms: npt.NDArray[np.floating], fwhm_kms: float) -> npt.NDArray[np.floating]:
    if fwhm_kms is None or fwhm_kms <= 0:
        k = np.zeros(1, dtype=float)
        k[0] = 1.0
        return k
    sigma = float(fwhm_kms) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    g = np.exp(-0.5 * (dv_kms / sigma) ** 2)
    # normalise to unit area in dv
    area = np.trapz(g, dv_kms)
    if area > 0:
        g /= area
    return g

def _rotational_kernel_gray(dv_kms: npt.NDArray[np.floating], vsini_kms: float, epsilon: float = 0.6) -> npt.NDArray[np.floating]:
    """Gray rotational broadening with linear limb darkening epsilon."""
    if vsini_kms is None or vsini_kms <= 0:
        k = np.zeros(1, dtype=float); k[0] = 1.0
        return k
    x = np.clip(dv_kms / float(vsini_kms), -1.0, 1.0)
    k = np.zeros_like(dv_kms, dtype=float)
    mask = np.abs(x) < 1.0 + 1e-12
    xin = x[mask]
    # Gray 2005 eqn for linear LD
    k[mask] = (2*(1-epsilon)*np.sqrt(1-xin**2) + (np.pi*epsilon/2.0)*(1-xin**2))
    area = np.trapz(k, dv_kms)
    if area > 0:
        k /= area
    return k

def _rt_kernel_gray(dv_kms: npt.NDArray[np.floating],
                    vmacro_fwhm_kms: float,
                    epsilon: float = 0.6,
                    frac_radial: float = 0.5,
                    n_mu: int = 40,
                    n_phi: int = 40) -> npt.NDArray[np.floating]:
    """Gray radial–tangential macroturbulence kernel via numerical μ,φ integration.

    We interpret `vmacro_fwhm_kms` as the *1D FWHM* of the Gaussian velocity field (ζ_RT).
    Radial component LOS sigma = ζ * μ; tangential LOS sigma = ζ * sqrt(1-μ^2) * |cosφ|.
    Limb darkening: I(μ) = 1 - ε + ε μ; surface element weight ∝ I(μ) μ dμ dφ.
    Radial/tangential mixture is set by `frac_radial` (default 0.5/0.5).
    """
    if vmacro_fwhm_kms is None or vmacro_fwhm_kms <= 0:
        k = np.zeros(1, dtype=float); k[0] = 1.0
        return k

    zeta_sigma = float(vmacro_fwhm_kms) / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # km/s

    # Integration grids
    mu = np.linspace(0.0, 1.0, n_mu)
    # Avoid exact zeros to prevent sigma=0
    mu[0] = 1e-6
    phi = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)

    # limb darkening weight * projected area
    I_mu = (1.0 - epsilon) + epsilon * mu
    w_mu = I_mu * mu
    w_mu /= np.trapz(w_mu, mu)  # normalise μ-weights

    k = np.zeros_like(dv_kms, dtype=float)
    for i, mui in enumerate(mu):
        w_i = w_mu[i]
        # --- radial part ---
        sig_r = zeta_sigma * mui
        if sig_r < 1e-8:
            # delta-function contribution
            kr = np.zeros_like(dv_kms); kr[np.argmin(np.abs(dv_kms))] = 1.0
        else:
            kr = np.exp(-0.5 * (dv_kms / sig_r) ** 2) / (np.sqrt(2*np.pi) * sig_r)

        # --- tangential part --- (average over phi)
        kt = np.zeros_like(dv_kms)
        for ph in phi:
            sig_t = zeta_sigma * np.sqrt(max(0.0, 1.0 - mui**2)) * abs(np.cos(ph))
            if sig_t < 1e-8:
                kt_phi = np.zeros_like(dv_kms); kt_phi[np.argmin(np.abs(dv_kms))] = 1.0
            else:
                kt_phi = np.exp(-0.5 * (dv_kms / sig_t) ** 2) / (np.sqrt(2*np.pi) * sig_t)
            kt += kt_phi
        kt /= len(phi)

        k += w_i * (frac_radial * kr + (1.0 - frac_radial) * kt)

    # Normalise kernel
    area = np.trapz(k, dv_kms)
    if area > 0:
        k /= area
    return k

def _safe_convolve_same(y: npt.NDArray[np.floating], k: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    k = np.asarray(k, float)
    if k.ndim != 1:
        k = k.ravel()
    s = k.sum()
    if s != 0:
        k = k / s
    return np.convolve(y, k, mode='same')

def _chunk_edges(x: npt.NDArray[np.floating], step_angstrom: float) -> npt.NDArray[np.int_]:
    edges = [0]
    x0 = float(x[0])
    for i in range(1, len(x)):
        if x[i] - x0 >= step_angstrom:
            edges.append(i)
            x0 = float(x[i])
    edges.append(len(x))
    return np.unique(np.asarray(edges, dtype=int))

# -----------------------------
# Main function
# -----------------------------

@overload
def convol_by_steps(
    wavelength_input: npt.NDArray[np.floating],
    flux_input: npt.NDArray[np.floating],
    *,
    vsini: Optional[float] = ...,
    resol: Optional[float] = ...,
    vdop: Optional[float] = ...,
    ainst: Optional[float] = ...,
    vmacro: Optional[float] = ...,
    limb_darkening: float = ...,
    rt_frac_radial: float = ...,
    step_angstrom: float = ...,
    return_kernel: Literal[True]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

@overload
def convol_by_steps(
    wavelength_input: npt.NDArray[np.floating],
    flux_input: npt.NDArray[np.floating],
    *,
    vsini: Optional[float] = ...,
    resol: Optional[float] = ...,
    vdop: Optional[float] = ...,
    ainst: Optional[float] = ...,
    vmacro: Optional[float] = ...,
    limb_darkening: float = ...,
    rt_frac_radial: float = ...,
    step_angstrom: float = ...,
    return_kernel: Literal[False] = ...
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

def convol_by_steps(
    wavelength_input: npt.NDArray[np.floating],
    flux_input: npt.NDArray[np.floating],
    *,
    vsini: Optional[float] = None,
    resol: Optional[float] = None,
    vdop: Optional[float] = None,
    ainst: Optional[float] = None,
    vmacro: Optional[float] = None,
    limb_darkening: float = 0.6,
    rt_frac_radial: float = 0.5,
    step_angstrom: float = 100.0,
    return_kernel: bool = False
):
    """Convolve a spectrum in ~step_angstrom windows with chosen broadening.

    Mutually exclusive choices for the non-rotational term (choose one): `resol`, `vdop`, `ainst`, `vmacro`.
    `vsini` rotational broadening is always applied (if >0) and effectively first.
    Units: Å for wavelengths, km/s for velocities and FWHM.
    """
    x = np.asarray(wavelength_input, dtype=float)
    y = np.asarray(flux_input, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("wavelength_input and flux_input must be 1D arrays of equal length")

    # Enforce mutual exclusivity
    picks = sum(int(v is not None and v > 0) for v in (resol, vdop, ainst, vmacro))
    if picks > 1:
        raise ValueError("Choose exactly one of: resol, vdop, ainst, vmacro")

    # Ensure increasing order
    if x[0] > x[-1]:
        x = x[::-1].copy()
        y = y[::-1].copy()

    f_out = np.zeros_like(y)
    edges = _chunk_edges(x, step_angstrom=step_angstrom)

    last_dv = None
    last_k = None

    for i in range(len(edges) - 1):
        i0, i1 = int(edges[i]), int(edges[i+1])
        if i1 - i0 < 5:
            f_out[i0:i1] = y[i0:i1]
            continue

        x_chunk = x[i0:i1]
        y_chunk = y[i0:i1]

        dl = np.median(np.diff(x_chunk))
        lam0 = 0.5 * (x_chunk[0] + x_chunk[-1])
        dv_sample = (dl / lam0) * C_KMS

        # Determine the broadening FWHM in velocity if Gaussian options are used
        fwhm_gauss_v = None
        if vdop and vdop > 0:
            fwhm_gauss_v = float(vdop)
        elif resol and resol > 0:
            fwhm_gauss_v = C_KMS / float(resol)
        elif ainst and ainst > 0:
            fwhm_gauss_v = C_KMS * (float(ainst) / lam0)

        # Build a dv grid wide enough to cover all kernels
        vmaxs = []
        if vsini and vsini > 0: vmaxs.append(float(vsini))
        if fwhm_gauss_v:  # Gaussian-based
            sigma = fwhm_gauss_v / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            vmaxs.append(5.0 * sigma)
        if vmacro and vmacro > 0:  # RT has wings comparable to Gaussian scale
            sigma_rt = float(vmacro) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            vmaxs.append(5.0 * sigma_rt)
        vmax = max(vmaxs) if vmaxs else 0.0
        vmax = max(vmax, 3.0 * dv_sample)
        nvel = int(np.ceil(2 * vmax / dv_sample)) | 1
        dv_kms = np.linspace(-vmax, vmax, nvel)

        # Compose kernel in velocity space
        k_total = np.zeros_like(dv_kms)
        k_total[np.argmin(np.abs(dv_kms))] = 1.0  # delta

        # Rotational first (conceptually)
        k_rot = _rotational_kernel_gray(dv_kms, vsini or 0.0, epsilon=limb_darkening)
        if len(k_rot) > 1:
            k_total = _safe_convolve_same(k_total, k_rot)

        # Then the mutually-exclusive choice
        if fwhm_gauss_v:
            k_gauss = _gaussian_kernel_velocity(dv_kms, fwhm_kms=fwhm_gauss_v)
            k_total = _safe_convolve_same(k_total, k_gauss)
        elif vmacro and vmacro > 0:
            k_rt = _rt_kernel_gray(dv_kms, vmacro_fwhm_kms=float(vmacro),
                                   epsilon=limb_darkening, frac_radial=rt_frac_radial)
            k_total = _safe_convolve_same(k_total, k_rt)

        # Normalize just in case
        area = np.trapz(k_total, dv_kms)
        if area > 0:
            k_total /= area

        # Map velocity kernel to wavelength grid at lam0
        dlam = (dv_kms / C_KMS) * lam0
        lam_kernel = lam0 + dlam
        k_pix = np.interp(x_chunk, lam_kernel, k_total, left=0.0, right=0.0)
        s = np.trapz(k_pix, x_chunk)
        if s > 0:
            k_pix /= s

        f_conv = _safe_convolve_same(y_chunk, k_pix)
        f_out[i0:i1] = f_conv

        last_dv = dv_kms
        last_k = k_total

    if return_kernel:
        return x, f_out, last_dv, last_k
    return x, f_out

# -----------------------------
# Demo
# -----------------------------
def _demo():
    import matplotlib.pyplot as plt
    # Simple synthetic spectrum with a few lines
    def make_spec(l0=5000.0, l1=5200.0, n=6000):
        lam = np.linspace(l0, l1, n)
        flux = np.ones_like(lam)
        for centre, depth, fwhm in [(5060.0, 0.6, 0.3), (5110.0, 0.7, 0.2), (5180.0, 0.5, 0.25)]:
            sigma = fwhm / (2*np.sqrt(2*np.log(2)))
            flux *= (1 - depth*np.exp(-0.5*((lam-centre)/sigma)**2))
        return lam, flux

    w, f = make_spec()
    w_out, f_out = convol_by_steps(w, f, vsini=20.0, resol=60000.0)

    plt.figure(figsize=(9,3.5))
    plt.plot(w, f, label="Original")
    plt.plot(w_out, f_out, label="Broadened")
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [arb]")
    plt.title("Convolution by steps (Python) — fixed normalisation")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    _demo()
