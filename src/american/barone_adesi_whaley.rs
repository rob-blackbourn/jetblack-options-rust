//! Option pricing functions implementing the Barone, Adesi and Whaley (1987)
//!
//! American approximation.
//!
//! The following arguments are common.
//!
//! * K (f64): The strike.
//! * T (f64): The time to expiry in years.
//! * r (f64): The risk free rate.
//! * b (f64): The asset growth.
//! * v (f64): The volatility.
//! * max_iterations (usize): The maximum number of iterations before a price is returned.
//! * epsilon (f64): The largest acceptable error.

use libm::{exp, fabs, log, pow, sqrt};

use crate::european::GeneralizedBlackScholes as BS;
use crate::{fdm::FdmWithCarry, implied_volatility::solve_ivol};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}

fn sqr(x: f64) -> f64 {
    x * x
}

pub struct BaroneAdesiWhaley {}

impl BaroneAdesiWhaley {
    /// Newton Raphson algorithm to solve for the critical commodity price for a call.
    #[allow(non_snake_case)]
    fn _kc(K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
        // Calculate the seed value Si
        let n = 2.0 * b / (v * v);
        let m = 2.0 * r / (v * v);
        let q2u = (-(n - 1.0) + sqrt(sqr(n - 1.0) + 4.0 * m)) / 2.0;
        let su = K / (1.0 - 1.0 / q2u);
        let h2 = -(b * T + 2.0 * v * sqrt(T)) * K / (su - K);
        let mut Si = K + (su - K) * (1.0 - exp(h2));

        let k = 2.0 * r / ((v * v) * (1.0 - exp(-r * T)));
        let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
        let q2 = (-(n - 1.0) + sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
        let mut lhs = Si - K;
        let mut rhs =
            BS::price(true, Si, K, T, r, b, v) + (1.0 - exp((b - r) * T) * cdf(d1)) * Si / q2;
        let mut bi = exp((b - r) * T) * cdf(d1) * (1.0 - 1.0 / q2)
            + (1.0 - exp((b - r) * T) * cdf(d1) / (v * sqrt(T))) / q2;
        let epsilon = 0.000001;
        // Using the Newton Raphson algorithm solve for Si
        while fabs(lhs - rhs) / K > epsilon {
            Si = (K + rhs - bi * Si) / (1.0 - bi);
            let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
            lhs = Si - K;
            rhs = BS::price(true, Si, K, T, r, b, v) + (1.0 - exp((b - r) * T) * cdf(d1)) * Si / q2;
            bi = exp((b - r) * T) * cdf(d1) * (1.0 - 1.0 / q2)
                + (1.0 - exp((b - r) * T) * pdf(d1) / (v * sqrt(T))) / q2;
        }

        Si
    }

    #[allow(non_snake_case)]
    fn _call_price(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
        if b >= r {
            return BS::price(true, S, K, T, r, b, v);
        }

        let Sk = BaroneAdesiWhaley::_kc(K, T, r, b, v);
        let n = 2.0 * b / (v * v);
        let k = 2.0 * r / ((v * v) * (1.0 - exp(-r * T)));
        let d1 = (log(Sk / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
        let q2 = (-(n - 1.0) + sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
        let a2 = (Sk / q2) * (1.0 - exp((b - r) * T) * cdf(d1));
        if S < Sk {
            BS::price(true, S, K, T, r, b, v) + a2 * pow(S / Sk, q2)
        } else {
            S - K
        }
    }

    /// Newton Raphson algorithm to solve for the critical commodity price for a put.
    #[allow(non_snake_case)]
    fn _kp(K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
        // Calculation of seed value, Si
        let n = 2.0 * b / (v * v);
        let m = 2.0 * r / (v * v);
        let q1u = (-(n - 1.0) - sqrt(sqr(n - 1.0) + 4.0 * m)) / 2.0;
        let su = K / (1.0 - 1.0 / q1u);
        let h1 = (b * T - 2.0 * v * sqrt(T)) * K / (K - su);
        let mut Si = su + (K - su) * exp(h1);

        let k = 2.0 * r / (v * 2.0 * (1.0 - exp(-r * T)));
        let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
        let q1 = (-(n - 1.0) - sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
        let mut lhs = K - Si;
        let mut rhs =
            BS::price(false, Si, K, T, r, b, v) - (1.0 - exp((b - r) * T) * cdf(-d1)) * Si / q1;
        let mut bi = -exp((b - r) * T) * cdf(-d1) * (1.0 - 1.0 / q1)
            - (1.0 + exp((b - r) * T) * pdf(-d1) / (v * sqrt(T))) / q1;
        let epsilon = 0.000001;
        // Using the Newton Raphson algorithm, solve for Si.
        while fabs(lhs - rhs) / K > epsilon {
            Si = (K - rhs + bi * Si) / (1.0 + bi);
            let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
            lhs = K - Si;
            rhs =
                BS::price(false, Si, K, T, r, b, v) - (1.0 - exp((b - r) * T) * cdf(-d1)) * Si / q1;
            bi = -exp((b - r) * T) * cdf(-d1) * (1.0 - 1.0 / q1)
                - (1.0 + exp((b - r) * T) * cdf(-d1) / (v * sqrt(T))) / q1;
        }

        Si
    }

    #[allow(non_snake_case)]
    fn _put_price(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
        let Sk = BaroneAdesiWhaley::_kp(K, T, r, b, v);
        let n = 2.0 * b / (v * v);
        let k = 2.0 * r / ((v * v) * (1.0 - exp(-r * T)));
        let d1 = (log(Sk / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
        let q1 = (-(n - 1.0) - sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
        let a1 = -(Sk / q1) * (1.0 - exp((b - r) * T) * cdf(-d1));

        if S > Sk {
            BS::price(false, S, K, T, r, b, v) + a1 * pow(S / Sk, q1)
        } else {
            K - S
        }
    }

    /// The Barone-Adesi and Whaley (1987) American approximation.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
        if is_call {
            BaroneAdesiWhaley::_call_price(S, K, T, r, b, v)
        } else {
            BaroneAdesiWhaley::_put_price(S, K, T, r, b, v)
        }
    }

    /// Calculate the volatility of an option that is implied by the price.
    #[allow(non_snake_case)]
    pub fn ivol(
        is_call: bool,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        p: f64,
        max_iterations: usize,
        epsilon: f64,
    ) -> f64 {
        solve_ivol(
            p,
            |v| BaroneAdesiWhaley::price(is_call, S, K, T, r, b, v),
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks(is_call: bool) -> FdmWithCarry {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, T: f64, r: f64, b: f64, v: f64| {
            BaroneAdesiWhaley::price(is_call, S, K, T, r, b, v)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use libm::fabs;

    use crate::fdm::DifferenceMethod;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = fabs(actual - expected);
        diff < threshold
    }

    #[test]
    fn it_should_calc_price() {
        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.087510335081676,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.510639694796271,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8736244925135566,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.938715732901822,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.7892100659783038,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                9.484654220828427,
            ),
        ] {
            let b = r - q;
            let actual = BaroneAdesiWhaley::price(is_call, S, K, T, r, b, v);
            assert!(is_close_to(actual, expected, 1e-12))
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.8592614442108903,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.10482043749096559,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.541088570403403,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.42462427819012216,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17169091749034138,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.813090973112196,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].delta(S, K, T, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-12));
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.01870512376100919,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.01850032019357073,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042923920551274364,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.04360354515231535,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028399659561806345,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.03266594122308675,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].gamma(S, K, T, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-9));
        }
    }

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.6059936396929517,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.4788290399032804,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -4.067355373044116,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.284344775164011,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.48956425165973,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.11716003329799829,
            ),
        ] {
            let b = r - q;
            let numeric =
                ng[&is_call].theta(S, K, T, r, b, v, 1.0 / 365.0, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-11));
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                14.2656452419061,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.9833977312675,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.89826727142619,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.853425641593763,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.77822568607357,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                16.181880739890353,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].vega(S, K, T, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-11));
        }
    }

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                39.274542330740125,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -5.709938455815355,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                24.580558678367613,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -20.964770823997945,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                8.060271439361,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -34.82234069684775,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].rho(S, K, T, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-10));
        }
    }
}
