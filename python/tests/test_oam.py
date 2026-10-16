"""Physical-sign, sampling, optimization and export checks for the OAM pilot."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metasurface_designer import design_metasurface, evaluate_fitness, compute_steering_matrix
from oam import analyze_vortex, design_vortex, _coordinates, _field, STATE_WEIGHTS, C0


class VortexPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        cls.analyses = {}
        for charge in (-2, -1, 1, 2):
            result = design_vortex(charge=charge)
            cls.results[charge] = result
            cls.analyses[charge] = analyze_vortex(result['B'], charge, map_size=9)

    def test_hard_states_and_expected_signed_modes(self):
        for charge, result in self.results.items():
            with self.subTest(charge=charge):
                metrics = self.analyses[charge]['metrics']
                self.assertEqual(result['B'].shape, (36, 36))
                self.assertTrue(np.isin(result['B'], [0, 1, 2, 3]).all())
                self.assertEqual(metrics['dominant_charge'], charge)
                self.assertAlmostEqual(metrics['ring_winding'], charge, places=8)
                self.assertGreater(metrics['target_mode_fraction'], 0.8)
                self.assertLess(metrics['axis_to_peak_db'], -30)

    def test_actual_hard_code_objective_and_monotone_search(self):
        for charge, result in self.results.items():
            with self.subTest(charge=charge):
                analysis = analyze_vortex(result['B'], charge, radial_samples=20,
                                           angular_samples=96, map_size=9)
                m = analysis['metrics']
                objective = (m['target_mode_power'] - 0.5*(m['disk_power']-m['target_mode_power'])
                             - 0.1*m['axis_power'])
                self.assertAlmostEqual(objective, result['objective'], places=12)
                self.assertAlmostEqual(result['fitness'], -objective, places=12)
                self.assertTrue(np.all(np.diff(result['objective_trace']) >= -1e-12))
                self.assertGreater(result['objective'], result['initial_objective'])

    def test_parseval_and_radial_integral(self):
        analysis = self.analyses[1]
        radius = analysis['radial_grid']
        values = np.mean(np.abs(analysis['polar_field'])**2, axis=1)
        numerator = np.sum(np.diff(radius)*(values[:-1]*radius[:-1] + values[1:]*radius[1:])/2)
        denominator = radius[-1]**2/2
        self.assertAlmostEqual(numerator/denominator, analysis['metrics']['disk_power'], places=13)
        self.assertAlmostEqual(float(analysis['mode_fractions'].sum()), 1.0, places=13)

    def test_conjugate_charge_symmetry(self):
        for charge in (1, 2):
            self.assertTrue(np.array_equal(self.results[-charge]['B'], (-self.results[charge]['B']) % 4))
            self.assertAlmostEqual(self.analyses[charge]['metrics']['target_mode_fraction'],
                                   self.analyses[-charge]['metrics']['target_mode_fraction'], places=12)

    def test_denominator_includes_unshown_modes(self):
        analysis = self.analyses[1]
        self.assertEqual(len(analysis['mode_fractions']), 192)
        self.assertGreater(float(analysis['mode_fractions'][np.abs(analysis['mode_orders']) > 6].sum()), 0)
        i = np.flatnonzero(analysis['mode_orders'] == 1)[0]
        self.assertAlmostEqual(float(analysis['mode_fractions'][i]), analysis['metrics']['target_mode_fraction'])

    def test_denser_sampling_agrees(self):
        dense = analyze_vortex(self.results[1]['B'], 1, radial_samples=88, angular_samples=384, map_size=9)
        self.assertLess(abs(dense['metrics']['target_mode_fraction']-
                            self.analyses[1]['metrics']['target_mode_fraction']), 0.005)
        self.assertAlmostEqual(dense['metrics']['ring_winding'], 1.0, places=8)

    def test_plane_wave_is_a_negative_control(self):
        control = analyze_vortex(np.zeros((36, 36), dtype=int), 1,
                                  radial_samples=24, angular_samples=128, map_size=9)
        self.assertEqual(control['metrics']['dominant_charge'], 0)
        self.assertLess(control['metrics']['target_mode_fraction'], 1e-8)
        self.assertFalse(control['metrics']['ring_phase_valid'])
        self.assertIsNone(control['metrics']['ring_winding'])

    def test_centered_field_matches_direct_element_sum(self):
        B = self.results[1]['B']
        rho = 1.7e12*70e-6/C0
        x, y = _coordinates(36, 36, rho)
        u, v = np.array([0.0, 0.023, -0.081]), np.array([0.0, 0.07, 0.014])
        field = _field(STATE_WEIGHTS[B.ravel()], x, y, u, v)
        direct = []
        for uu, vv in zip(u, v):
            value = 0j
            for m in range(36):
                for n in range(36):
                    xx=(m-17.5)*rho
                    yy=(n-17.5)*rho
                    value += np.exp(-0.5j*np.pi*B[m,n])*np.exp(-2j*np.pi*(xx*uu+yy*vv))
            direct.append(value/B.size)
        np.testing.assert_allclose(field, direct, atol=2e-14, rtol=1e-12)

    def test_seed_is_reproducible_and_does_not_change_global_rng(self):
        np.random.seed(1337)
        before = np.random.get_state()
        a = design_vortex(N=16, seed=19)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])
        b = design_vortex(N=16, seed=19)
        self.assertTrue(np.array_equal(a['B'], b['B']))

    def test_sidecars_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'vortex.txt'
            result=design_metasurface(N=16, beam_type='vortex', oam_charge=-1, output_path=output)
            np.testing.assert_array_equal(np.loadtxt(output,delimiter=',',dtype=int), result['B'])
            metadata=json.loads(Path(result['oam_files']['metrics']).read_text(encoding='utf-8'))
            self.assertEqual(metadata['target_charge'], -1)
            with np.load(result['oam_files']['fields']) as data:
                np.testing.assert_array_equal(data['codes'], result['B'])
                np.testing.assert_array_equal(data['initial_codes'], result['initial_B'])
                np.testing.assert_allclose(data['complex_map'], result['oam_analysis']['complex_map'], equal_nan=True)
            with Image.open(result['oam_files']['figure']) as picture:
                self.assertEqual(picture.size,(1240,850))

    def test_ordinary_beam_path_still_works(self):
        for bit in (1,2):
            a=design_metasurface(N=12,theta=[25,40],phi=[10,170],nbit=bit,method='analytical')
            b=design_metasurface(N=12,theta=[25,40],phi=[10,170],nbit=bit,method='fast')
            W=compute_steering_matrix(12,[25,40],[10,170])
            self.assertLessEqual(b['fitness'],a['fitness']+1e-10)
            self.assertAlmostEqual(b['fitness'],evaluate_fitness(b['B'],W,nbit=bit),places=9)
            self.assertNotIn('oam_analysis',b)

    def test_invalid_or_unimplemented_vortex_requests(self):
        for extra in [dict(oam_charge=0),dict(oam_charge=3),dict(oam_charge=1.5),
                      dict(nbit=1),dict(N=35),dict(N=8),dict(N=128),dict(N=36,M=32),
                      dict(theta=[10]),dict(theta=[0,0],phi=[0,180]),
                      dict(period_um=150),dict(period_um=4),dict(method='ga_exact')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                design_metasurface(beam_type='vortex',**extra)

    def test_wide_angle_diagnostic_exposes_observation_window_dependence(self):
        for charge, bounds in ((1, (0.755, 0.760)), (2, (0.744, 0.748))):
            analysis = self.analyses[charge]
            metrics = analysis['metrics']
            wide = metrics['wide_angle']
            self.assertEqual(wide['uv_radius'], 0.99)
            self.assertGreater(wide['target_mode_fraction'], bounds[0])
            self.assertLess(wide['target_mode_fraction'], bounds[1])
            self.assertLess(wide['target_mode_fraction'], metrics['target_mode_fraction'])
            self.assertAlmostEqual(float(analysis['wide_mode_fractions'].sum()), 1.0, places=13)
            expected=metrics['disk_power']*metrics['uv_radius']**2/(wide['disk_power']*0.99**2)
            self.assertAlmostEqual(wide['roi_disk_integral_fraction_estimate'], expected, places=13)
            self.assertFalse(wide['is_total_radiation_efficiency'])

    def test_ring_winding_does_not_claim_centered_singularity(self):
        for charge in (1, 2):
            m=self.analyses[charge]['metrics']
            self.assertAlmostEqual(m['ring_winding'], charge)
            self.assertFalse(m['axis_is_numerical_null'])
            self.assertFalse(m['on_axis_singularity_verified'])
            initial=analyze_vortex(self.results[charge]['initial_B'], charge, map_size=9,
                                   include_wide_angle=False)
            self.assertTrue(initial['metrics']['axis_is_numerical_null'])
            self.assertFalse(initial['metrics']['on_axis_singularity_verified'])

    def test_field_matches_independent_fft_with_centering_factor(self):
        rng=np.random.default_rng(27)
        B=rng.integers(0,4,(16,16));rho=.397;nfft=256
        F=np.fft.fft2(np.exp(-.5j*np.pi*B),s=(nfft,nfft))/B.size
        ij=rng.integers(0,nfft,(40,2));freq=np.fft.fftfreq(nfft,d=rho)
        u=freq[ij[:,0]];v=freq[ij[:,1]]
        expected=F[ij[:,0],ij[:,1]]*np.exp(2j*np.pi*(15*rho/2)*(u+v))
        x,y=_coordinates(16,16,rho)
        actual=_field(STATE_WEIGHTS[B.ravel()],x,y,u,v)
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-11)

    def test_sampling_input_validation_and_alias_flag(self):
        for extra in (dict(radial_samples=44.5),dict(angular_samples=True),dict(map_size=9.0)):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                analyze_vortex(self.results[1]['B'],1,**extra)
        coarse=analyze_vortex(self.results[1]['B'],1,uv_radius=.99,
                               angular_samples=32,map_size=9,include_wide_angle=False)
        self.assertFalse(coarse['metrics']['angular_sampling_sufficient'])
        self.assertIsNone(coarse['metrics']['ring_winding'])


if __name__=='__main__':
    unittest.main()
