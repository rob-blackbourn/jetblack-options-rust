//! # Option valuation functions implementing the Bjerksund and Stensland (2002)
//!
//! American approximation
//!
//! The following arguments are common:
//!
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The current asset price.
//! * K (f64): The option strike price
//! * t (f64): The time to maturity of the option in years.
//! * r (f64): The risk free rate.
//! * b (f64): The cost of carry of the asset.
//! * v (f64): The volatility of the asset.
//! * p (f64): The option price.
//! * max_iterations (usize): The maximum number of iterations before a price is returned.
//! * epsilon (f64): The largest acceptable error.

use crate::distributions::cbnd::cbnd;
use crate::distributions::cdf;
use crate::european::GeneralizedBlackScholes as BS;
use crate::fdm::FdmWithCarry;
use crate::implied_volatility::solve_ivol;

fn sqr(x: f64) -> f64 {
    x * x
}

pub struct BjerksundStensland2002 {}

impl BjerksundStensland2002 {
    #[allow(non_snake_case)]
    fn _phi(S: f64, t: f64, gamma_: f64, h: f64, i: f64, r: f64, b: f64, v: f64) -> f64 {
        let lambda_ = (-r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1.0) * (v * v)) * t;
        let d = -((S / h).ln() + (b + (gamma_ - 0.5) * (v * v)) * t) / (v * t.sqrt());
        let kappa = 2.0 * b / (v * v) + 2.0 * gamma_ - 1.0;
        f64::exp(lambda_)
            * f64::powf(S, gamma_)
            * (cdf(d) - (i / S).powf(kappa) * cdf(d - 2.0 * (i / S).ln() / (v * t.sqrt())))
    }

    #[allow(non_snake_case)]
    fn _ksi(
        S: f64,
        t2: f64,
        gamma_: f64,
        h: f64,
        I2: f64,
        I1: f64,
        t1: f64,
        r: f64,
        b: f64,
        v: f64,
    ) -> f64 {
        let e1 = ((S / I1).ln() + (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * t1.sqrt());
        let e2 =
            (((I2 * I2) / (S * I1)).ln() + (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * t1.sqrt());
        let e3 = ((S / I1).ln() - (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * t1.sqrt());
        let e4 =
            ((I2 * I2 / (S * I1)).ln() - (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * t1.sqrt());

        let f1 = ((S / h).ln() + (b + (gamma_ - 0.5) * (v * v)) * t2) / (v * t2.sqrt());
        let f2 =
            (((I2 * I2) / (S * h)).ln() + (b + (gamma_ - 0.5) * (v * v)) * t2) / (v * t2.sqrt());
        let f3 =
            (((I1 * I1) / (S * h)).ln() + (b + (gamma_ - 0.5) * (v * v)) * t2) / (v * t2.sqrt());
        let f4 = ((S * (I1 * I1) / (h * (I2 * I2))).ln() + (b + (gamma_ - 0.5) * (v * v)) * t2)
            / (v * t2.sqrt());

        let rho = (t1 / t2).sqrt();
        let lambda_ = -r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1.0) * (v * v);
        let kappa = 2.0 * b / (v * v) + (2.0 * gamma_ - 1.0);

        (lambda_ * t2).exp()
            * f64::powf(S, gamma_)
            * (cbnd(-e1, -f1, rho)
                - (I2 / S).powf(kappa) * cbnd(-e2, -f2, rho)
                - (I1 / S).powf(kappa) * cbnd(-e3, -f3, -rho)
                + (I1 / I2).powf(kappa) * cbnd(-e4, -f4, -rho))
    }

    #[allow(non_snake_case)]
    fn _call_price(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let t1 = 1.0 / 2.0 * (f64::sqrt(5.0) - 1.0) * t;

        if b >= r {
            // Use Black-Scholes as it is never optimal to exercise before maturity.
            return BS::price(true, S, K, t, r, b, v);
        }

        let beta =
            (1.0 / 2.0 - b / (v * v)) + (sqr(b / (v * v) - 1.0 / 2.0) + 2.0 * r / (v * v)).sqrt();
        let b_infinity = beta / (beta - 1.0) * K;
        let b0 = f64::max(K, r / (r - b) * K);

        let ht1 = -(b * t1 + 2.0 * v * t1.sqrt()) * (K * K) / ((b_infinity - b0) * b0);
        let ht2 = -(b * t + 2.0 * v * t.sqrt()) * (K * K) / ((b_infinity - b0) * b0);
        let I1 = b0 + (b_infinity - b0) * (1.0 - f64::exp(ht1));
        let I2 = b0 + (b_infinity - b0) * (1.0 - f64::exp(ht2));
        let alfa1 = (I1 - K) * f64::powf(I1, -beta);
        let alfa2 = (I2 - K) * f64::powf(I2, -beta);

        if S >= I2 {
            S - K
        } else {
            return alfa2 * f64::powf(S, beta)
                - alfa2 * BjerksundStensland2002::_phi(S, t1, beta, I2, I2, r, b, v)
                + BjerksundStensland2002::_phi(S, t1, 1.0, I2, I2, r, b, v)
                - BjerksundStensland2002::_phi(S, t1, 1.0, I1, I2, r, b, v)
                - K * BjerksundStensland2002::_phi(S, t1, 0.0, I2, I2, r, b, v)
                + K * BjerksundStensland2002::_phi(S, t1, 0.0, I1, I2, r, b, v)
                + alfa1 * BjerksundStensland2002::_phi(S, t1, beta, I1, I2, r, b, v)
                - alfa1 * BjerksundStensland2002::_ksi(S, t, beta, I1, I2, I1, t1, r, b, v)
                + BjerksundStensland2002::_ksi(S, t, 1.0, I1, I2, I1, t1, r, b, v)
                - BjerksundStensland2002::_ksi(S, t, 1.0, K, I2, I1, t1, r, b, v)
                - K * BjerksundStensland2002::_ksi(S, t, 0.0, I1, I2, I1, t1, r, b, v)
                + K * BjerksundStensland2002::_ksi(S, t, 0.0, K, I2, I1, t1, r, b, v);
        }
    }

    /// The Bjerksund and Stensland (2002) American approximation.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if is_call {
            BjerksundStensland2002::_call_price(S, K, t, r, b, v)
        } else {
            // Use the Bjerksund and Stensland put-call transformation
            BjerksundStensland2002::_call_price(K, S, t, r - b, -b, v)
        }
    }

    /// Calculate the volatility of an option that is implied by the price.
    #[allow(non_snake_case)]
    pub fn ivol(
        is_call: bool,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        b: f64,
        p: f64,
        max_iterations: usize,
        epsilon: f64,
    ) -> f64 {
        solve_ivol(
            p,
            |v| BjerksundStensland2002::price(is_call, S, K, t, r, b, v),
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks(is_call: bool) -> FdmWithCarry {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            BjerksundStensland2002::price(is_call, S, K, t, r, b, v)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::fdm::DifferenceMethod;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = (actual - expected).abs();
        diff < threshold
    }

    #[test]
    fn it_should_calc_price() {
        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.071608152504766,
                1e-12,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5121325794818574,
                1e-12,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8695471290130214,
                1e-12,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.013317901457917,
                1e-12,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.7881689776961025,
                1e-12,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                10.087984869664105,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = BjerksundStensland2002::price(is_call, S, K, t, r, b, v);

            assert!(
                is_close_to(actual, expected, threshold),
                "price({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, BjerksundStensland2002::fdm_greeks(true)),
            (false, BjerksundStensland2002::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.8573873104591456,
                1e-12,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.1060355803375046,
                1e-12,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5404729344355985,
                1e-10,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.44432535376159876,
                1e-12,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17153030785976853,
                1e-11,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.9106399684526423,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].delta(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(actual, expected, threshold),
                "[{}].delta({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, BjerksundStensland2002::fdm_greeks(true)),
            (false, BjerksundStensland2002::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018553562721734806,
                1e-9,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018941027661867338,
                1e-9,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042840733769367034,
                1e-9,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.04794256739160119,
                1e-9,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.02837656552401313,
                1e-9,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.04379615347716026,
                1e-9,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].gamma(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            let diff = (expected - actual).abs();
            assert!(
                diff < threshold,
                "[{}].gamma({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e}",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([
            (true, BjerksundStensland2002::fdm_greeks(true)),
            (false, BjerksundStensland2002::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.5309937384477044,
                1e-12,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.497522525972883,
                1e-11,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -4.040856526623422,
                1e-11,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.4950150852994923,
                1e-11,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.4811531657213237,
                1e-10,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.5957097928179778,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual =
                ng[&is_call].theta(S, K, t, r, b, v, 1.0 / 365.0, DifferenceMethod::Central);
            assert!(
                (expected - actual).abs() < threshold,
                "[{}]theta({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([
            (true, BjerksundStensland2002::fdm_greeks(true)),
            (false, BjerksundStensland2002::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.9782779912494,
                1e-11,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                14.07649727730842,
                1e-10,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.773915325762232,
                1e-10,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.939020685002646,
                1e-11,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.734971740807737,
                1e-10,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                9.059857617765843,
                1e-11,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].vega(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(actual, expected, threshold),
                "[{}].vega({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([
            (true, BjerksundStensland2002::fdm_greeks(true)),
            (false, BjerksundStensland2002::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                40.97350888627194,
                1e-10,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -5.692145861019071,
                1e-12,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                25.06962040216365,
                1e-11,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -18.115786191373218,
                1e-11,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                8.1822713879518,
                1e-10,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -10.273446554156962,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].rho(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(actual, expected, threshold),
                "[{}].rho({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }
}
