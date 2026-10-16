"""On-axis 2-bit OAM synthesis and sampled complex-field diagnostics.

Array rows are x, columns are y. Aperture coordinates are centered at zero.
Positive charge means exp(+j*l*azimuth) in the aperture and far-field ring.
Mode fractions are integrated over a specified uv disk, not total efficiency.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

C0 = 1.0 / np.sqrt(8.854187817e-12 * 4.0 * np.pi * 1e-7)
STATE_WEIGHTS = np.exp(-0.5j * np.pi * np.arange(4))
SUPPORTED_CHARGES = (-2, -1, 1, 2)


def validate_vortex_geometry(N, M, nbit, charge, frequency_thz, period_um):
    for name, value in (('N', N), ('M', M)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 16 <= value <= 64 or value % 2:
            raise ValueError(f"涡旋原型的 {name} 需为16到64之间的偶数")
    if N != M:
        raise ValueError("涡旋原型目前使用正方形阵面")
    if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, (int, np.integer)) or charge not in SUPPORTED_CHARGES:
        raise ValueError("涡旋拓扑荷目前支持 -2、-1、1、2")
    if nbit != 2:
        raise ValueError("涡旋原型目前仅支持2-bit编码")
    for value in (frequency_thz, period_um):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("频率与周期必须为有限正数")
    rho = float(frequency_thz) * 1e12 * float(period_um) * 1e-6 / C0
    if rho > 0.5 + 1e-12:
        raise ValueError("涡旋原型要求单元周期不大于半波长，请减小周期或频率")
    if N * rho < 4.0:
        raise ValueError("涡旋原型要求阵面边长至少为4个波长，请增大阵列或周期")
    return rho


def _codes(B):
    B = np.asarray(B)
    if B.ndim != 2 or not np.isin(B, [0, 1, 2, 3]).all():
        raise ValueError("涡旋评价需要二维2-bit整数状态矩阵")
    return B.astype(np.int32)


def _coordinates(N, M, rho):
    x = (np.arange(M) - (M - 1) / 2) * rho
    y = (np.arange(N) - (N - 1) / 2) * rho
    return np.meshgrid(x, y, indexing='ij')


def _radial_grid(radius, count):
    # Trapezoidal r*dr quadrature, normalized over the declared disk.
    r = np.linspace(0.0, radius, count)
    weights = r.copy()
    weights[-1] *= 0.5
    weights /= weights.sum()
    return r, weights


def _kernel(x, y, u, v):
    points = np.stack((np.asarray(u).ravel(), np.asarray(v).ravel()))
    coords = np.stack((x.ravel(), y.ravel()), axis=1)
    return np.exp(-2j * np.pi * (coords @ points)) / x.size


def _field(weights, x, y, u, v, chunk=512):
    # Separable rectangular-array sum; avoids a cells-by-directions allocation.
    u, v = np.broadcast_arrays(u, v)
    aperture = np.asarray(weights).reshape(x.shape)
    result = np.empty(u.size, dtype=np.complex128)
    for start in range(0, u.size, chunk):
        ux = np.exp(-2j*np.pi*np.outer(u.ravel()[start:start+chunk], x[:, 0]))
        vy = np.exp(-2j*np.pi*np.outer(v.ravel()[start:start+chunk], y[0, :]))
        result[start:start+chunk] = np.einsum('ki,ij,kj->k', ux, aperture, vy,
                                               optimize=True) / x.size
    return result.reshape(u.shape)


def _sample_spectrum(a, x, y, radius, radial_samples, angular_samples):
    r, radial_weights = _radial_grid(radius, radial_samples)
    psi = np.arange(angular_samples) * (2*np.pi/angular_samples)
    field = _field(a, x, y, r[:, None]*np.cos(psi), r[:, None]*np.sin(psi))
    coefficients = np.fft.fft(field, axis=1)/angular_samples
    orders = np.rint(np.fft.fftfreq(angular_samples, d=1.0/angular_samples)).astype(int)
    powers = np.sum(radial_weights[:, None]*np.abs(coefficients)**2, axis=0)
    return r, psi, field, orders, powers


def _powers(E, modal, axis, radial_weights):
    total = float(np.sum(radial_weights * np.mean(np.abs(E) ** 2, axis=1)))
    target = float(np.sum(radial_weights * np.abs(modal) ** 2))
    return target, max(0.0, total - target), float(np.abs(axis) ** 2)


def _score(E, modal, axis, radial_weights, leakage_weight, null_weight):
    target, leak, center = _powers(E, modal, axis, radial_weights)
    return target - leakage_weight * leak - null_weight * center


def design_vortex(N=36, M=None, charge=1, nbit=2, frequency_thz=1.7,
                  period_um=70.0, method='fast', max_passes=3, seed=42,
                  leakage_weight=0.5, null_weight=0.1):
    """Optimize target-mode energy minus leakage and on-axis field penalties.

The angular integral uses 20 radii and 96 azimuth samples. Each coordinate
change updates the full sampled field and its target-mode projection.
"""
    start = time.perf_counter()
    M = N if M is None else M
    rho = validate_vortex_geometry(N, M, nbit, charge, frequency_thz, period_um)
    if method not in ('fast', 'analytical'):
        raise ValueError("涡旋原型目前支持 fast 或 analytical 模式")
    if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)) or max_passes < 0:
        raise ValueError("max_passes 必须为非负整数")
    if not all(np.isfinite(w) and w >= 0 for w in (leakage_weight, null_weight)):
        raise ValueError("模式泄漏与轴心惩罚权重必须有限且非负")
    x, y = _coordinates(N, M, rho)
    azimuth = np.arctan2(y, x)
    # Reflection convention a=exp(-j*phase): phase=-l*azimuth.
    phase = -charge * azimuth
    B = np.rint(np.mod(phase, 2 * np.pi) / (np.pi / 2)).astype(np.int32) % 4
    initial_B = B.copy()
    radius = min(0.85, 4.0 / (N * rho))
    r, radial_weights = _radial_grid(radius, 20)
    psi = np.arange(96) * (2 * np.pi / 96)
    u = r[:, None] * np.cos(psi)
    v = r[:, None] * np.sin(psi)
    K = _kernel(x, y, u, v)
    K_polar = K.reshape(N * M, len(r), len(psi))
    L = np.mean(K_polar * np.exp(-1j * charge * psi)[None, None, :], axis=2)
    a = STATE_WEIGHTS[B.ravel()]
    E = (a @ K).reshape(len(r), len(psi))
    modal = a @ L
    axis = np.sum(a) / a.size
    initial_score = _score(E, modal, axis, radial_weights, leakage_weight, null_weight)
    trace = [initial_score]
    evals = 1
    accepted = 0
    visited_passes = 0
    if method == 'fast':
        rng = np.random.default_rng(seed)
        sample_weights = (radial_weights[:, None] / len(psi))
        total_diag = 1.0 / a.size ** 2
        for _ in range(max_passes):
            changed = False
            for j in rng.permutation(a.size):
                field_row = K_polar[j]
                mode_row = L[j]
                h_total = np.sum(np.conj(E) * field_row * sample_weights)
                h_target = np.sum(radial_weights * np.conj(modal) * mode_row)
                target_diag = np.sum(radial_weights * np.abs(mode_row) ** 2)
                h = ((1 + leakage_weight) * h_target - leakage_weight * h_total
                     - null_weight * np.conj(axis) / a.size)
                diagonal = ((1 + leakage_weight) * target_diag
                            - (leakage_weight + null_weight) * total_diag)
                delta = STATE_WEIGHTS - a[j]
                gains = 2 * np.real(delta * h) + np.abs(delta) ** 2 * diagonal
                state = int(np.argmax(gains))
                evals += 3
                if gains[state] > 1e-14:
                    change = delta[state]
                    E += change * field_row
                    modal += change * mode_row
                    axis += change / a.size
                    a[j] = STATE_WEIGHTS[state]
                    B.ravel()[j] = state
                    changed = True
                    accepted += 1
            visited_passes += 1
            trace.append(_score(E, modal, axis, radial_weights, leakage_weight, null_weight))
            if not changed:
                break
    # Recompute from final hard states; never report a drifted incremental score.
    a = STATE_WEIGHTS[B.ravel()]
    E = (a @ K).reshape(len(r), len(psi))
    modal = a @ L
    axis = np.mean(a)
    score = _score(E, modal, axis, radial_weights, leakage_weight, null_weight)
    target, leak, center = _powers(E, modal, axis, radial_weights)
    return dict(B=B, initial_B=initial_B, fitness=-score, elapsed_sec=time.perf_counter()-start,
                method='oam_' + method, beam_type='vortex', oam_charge=int(charge),
                evals=evals, accepted_changes=accepted, passes=visited_passes,
                initial_objective=initial_score, objective=score, objective_trace=trace,
                target_mode_power=target, unwanted_mode_power=leak, axis_power=center,
                observation_uv_radius=radius, leakage_weight=leakage_weight,
                null_weight=null_weight, objective_kind='sampled_oam_energy_minus_leakage',
                optimization_radial_samples=len(r), optimization_angular_samples=len(psi))


def analyze_vortex(B, charge, frequency_thz=1.7, period_um=70.0,
                   uv_radius=None, radial_samples=44, angular_samples=192, map_size=101,
                   include_wide_angle=True):
    """Denser polar resampling and a centered complex-field map.

The spectrum denominator includes every FFT bin. It is a conditional fraction
inside the declared uv disk with r*dr measure, not all-space OAM efficiency.
"""
    B = _codes(B)
    M, N = B.shape
    rho = validate_vortex_geometry(N, M, 2, charge, frequency_thz, period_um)
    for name, value in (('radial_samples', radial_samples), ('angular_samples', angular_samples),
                        ('map_size', map_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} 必须为整数")
    if radial_samples < 8 or angular_samples < 32 or angular_samples % 2:
        raise ValueError("模式评价至少需要8个半径及32个偶数方位采样")
    if abs(charge) >= angular_samples // 2:
        raise ValueError("方位采样不足以分辨目标模式")
    if map_size < 9 or map_size % 2 == 0:
        raise ValueError("场图尺寸应为不小于9的奇数")
    radius = min(0.85, 4.0 / (N * rho)) if uv_radius is None else float(uv_radius)
    if not np.isfinite(radius) or not 0 < radius < 1:
        raise ValueError("观测uv半径必须在0和1之间")
    x, y = _coordinates(N, M, rho)
    a = STATE_WEIGHTS[B.ravel()]
    r, psi, polar_field, orders, mode_powers = _sample_spectrum(
        a, x, y, radius, radial_samples, angular_samples)
    angular_sampling_sufficient = angular_samples >= int(np.ceil(2*np.pi*np.sqrt(2)*(N-1)*rho*radius+16))
    total_power = float(np.sum(mode_powers))
    if total_power <= 1e-20:
        raise ValueError("观测区域内复场能量过低，无法计算模式占比")
    fractions = mode_powers / total_power
    target_index = int(np.flatnonzero(orders == charge)[0])
    radial_power = np.mean(np.abs(polar_field) ** 2, axis=1)
    ring_index = int(np.argmax(radial_power))
    ring_field = polar_field[ring_index]
    ring_valid = bool(angular_sampling_sufficient and ring_index > 0
                      and np.min(np.abs(ring_field)) > 1e-4*np.max(np.abs(ring_field)))
    winding = (float(np.sum(np.angle(np.roll(ring_field, -1) * np.conj(ring_field))) / (2*np.pi))
               if ring_valid else None)
    axis_power = float(np.abs(np.mean(a)) ** 2)
    peak = float(np.max(np.abs(polar_field) ** 2))
    grid = np.linspace(-radius, radius, map_size)
    U, V = np.meshgrid(grid, grid, indexing='xy')
    valid = U*U + V*V <= radius*radius*(1+1e-12)
    complex_map = np.full(U.shape, np.nan + 1j*np.nan, dtype=np.complex128)
    complex_map[valid] = _field(a, x, y, U[valid], V[valid])
    metrics = dict(rows=M, cols=N, frequency_thz=float(frequency_thz), period_um=float(period_um),
                   target_charge=int(charge), dominant_charge=int(orders[np.argmax(fractions)]),
                   target_mode_fraction=float(fractions[target_index]),
                   target_mode_power=float(mode_powers[target_index]), disk_power=total_power,
                   axis_power=axis_power, sampled_polar_peak_power=peak,
                   axis_is_numerical_null=bool(axis_power <= peak*1e-24),
                   on_axis_singularity_verified=False,
                   axis_to_peak_db=float(10*np.log10(max(axis_power/max(peak,1e-30), 1e-15))),
                   ring_uv_radius=float(r[ring_index]), ring_winding=winding,
                   ring_phase_valid=ring_valid, uv_radius=radius,
                   radial_samples=int(radial_samples), angular_samples=int(angular_samples),
                   angular_sampling_sufficient=angular_sampling_sufficient,
                   spectrum_definition='centered uv disk; radial r*dr quadrature; all FFT modes in denominator',
                   is_full_wave=False)
    result = dict(metrics=metrics, mode_orders=orders, mode_powers=mode_powers,
                mode_fractions=fractions, radial_grid=r, radial_power=radial_power,
                ring_azimuth=psi, ring_field=ring_field, polar_field=polar_field,
                uv_grid=grid, complex_map=complex_map, valid_map=valid)
    if include_wide_angle:
        # A second, explicitly declared field of view exposes leakage outside the ROI.
        # Sampling scales with the electrical aperture, including at N=64, p=lambda/2.
        wide_radius = 0.99
        nr = max(96, int(np.ceil(8*N*rho*wide_radius))+1)
        required = max(256, int(np.ceil(2*np.pi*np.sqrt(2)*N*rho*wide_radius+32)))
        na = 1 << (required-1).bit_length()
        wr, wp, wf, wo, powers = _sample_spectrum(a, x, y, wide_radius, nr, na)
        wide_total = float(powers.sum())
        metrics['wide_angle'] = dict(uv_radius=wide_radius, radial_samples=nr,
            angular_samples=na, target_mode_fraction=float(powers[wo == charge].sum()/wide_total),
            disk_power=wide_total,
            roi_disk_integral_fraction_estimate=(float(total_power*radius**2/(wide_total*wide_radius**2))
                                       if radius <= wide_radius else None),
            is_total_radiation_efficiency=False)
        result.update(wide_mode_orders=wo, wide_mode_powers=powers,
                      wide_mode_fractions=powers/wide_total, wide_radial_grid=wr,
                      wide_azimuth=wp, wide_polar_field=wf)
    return result


def save_vortex_analysis(analysis, output_path, B=None, initial_B=None):
    """Save JSON metrics, reproducible complex fields and an optional panel PNG."""
    path = Path(os.fspath(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix('')
    json_path = Path(str(stem) + '_oam.json')
    npz_path = Path(str(stem) + '_oam.npz')
    json_path.write_text(json.dumps(analysis['metrics'], ensure_ascii=False, indent=2), encoding='utf-8')
    arrays = {key: value for key, value in analysis.items() if isinstance(value, np.ndarray)}
    if B is not None:
        arrays['codes'] = B
    if initial_B is not None:
        arrays['initial_codes'] = initial_B
    np.savez_compressed(npz_path, **arrays)
    result = dict(metrics=str(json_path), fields=str(npz_path))
    if B is not None:
        from oam_view import render_oam_panel
        png_path = Path(str(stem) + '_oam.png')
        render_oam_panel(B, analysis).save(png_path)
        result['figure'] = str(png_path)
    return result
