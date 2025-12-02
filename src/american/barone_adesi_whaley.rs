//! Option pricing functions implementing the Barone, Adesi and Whaley (1987)
//!
//! American approximation.
//!
//! The following arguments are common.
//!
//! * K (f64): The strike.
//! * t (f64): The time to expiry in years.
//! * r (f64): The risk free rate.
//! * b (f64): The asset growth.
//! * v (f64): The volatility.
//! * max_iterations (usize): The maximum number of iterations before a price is returned.
//! * epsilon (f64): The largest acceptable error.

use crate::distributions::{cdf, pdf};
use crate::european::GeneralizedBlackScholes as BS;
use crate::fdm::FdmWithCarry;
use crate::implied_volatility::solve_ivol;

fn sqr(x: f64) -> f64 {
    x * x
}

pub struct BaroneAdesiWhaley {}

impl BaroneAdesiWhaley {
    /// Newton Raphson algorithm to solve for the critical commodity price for a call.
    #[allow(non_snake_case)]
    fn _kc(K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        // Calculate the seed value Si
        let n = 2.0 * b / (v * v);
        let m = 2.0 * r / (v * v);
        let q2u = (-(n - 1.0) + (sqr(n - 1.0) + 4.0 * m.sqrt())) / 2.0;
        let su = K / (1.0 - 1.0 / q2u);
        let h2 = -(b * t + 2.0 * v * t.sqrt()) * K / (su - K);
        let mut Si = K + (su - K) * (1.0 - f64::exp(h2));

        let k = 2.0 * r / ((v * v) * (1.0 - (-r * t).exp()));
        let d1 = ((Si / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let q2 = (-(n - 1.0) + (sqr(n - 1.0) + 4.0 * k).sqrt()) / 2.0;
        let mut lhs = Si - K;
        let mut rhs =
            BS::price(true, Si, K, t, r, b, v) + (1.0 - ((b - r) * t).exp() * cdf(d1)) * Si / q2;
        let mut bi = ((b - r) * t).exp() * cdf(d1) * (1.0 - 1.0 / q2)
            + (1.0 - ((b - r) * t).exp() * cdf(d1) / (v * t.sqrt())) / q2;
        let epsilon = 0.000001;
        // Using the Newton Raphson algorithm solve for Si
        while f64::abs(lhs - rhs) / K > epsilon {
            Si = (K + rhs - bi * Si) / (1.0 - bi);
            let d1 = ((Si / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
            lhs = Si - K;
            rhs = BS::price(true, Si, K, t, r, b, v)
                + (1.0 - ((b - r) * t).exp() * cdf(d1)) * Si / q2;
            bi = ((b - r) * t).exp() * cdf(d1) * (1.0 - 1.0 / q2)
                + (1.0 - ((b - r) * t).exp() * pdf(d1) / (v * t.sqrt())) / q2;
        }

        Si
    }

    #[allow(non_snake_case)]
    fn _call_price(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if b >= r {
            return BS::price(true, S, K, t, r, b, v);
        }

        let Sk = BaroneAdesiWhaley::_kc(K, t, r, b, v);
        let n = 2.0 * b / (v * v);
        let k = 2.0 * r / ((v * v) * (1.0 - (-r * t).exp()));
        let d1 = ((Sk / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let q2 = (-(n - 1.0) + (sqr(n - 1.0) + 4.0 * k).sqrt()) / 2.0;
        let a2 = (Sk / q2) * (1.0 - ((b - r) * t).exp() * cdf(d1));
        if S < Sk {
            BS::price(true, S, K, t, r, b, v) + a2 * (S / Sk).powf(q2)
        } else {
            S - K
        }
    }

    /// Newton Raphson algorithm to solve for the critical commodity price for a put.
    #[allow(non_snake_case)]
    fn _kp(K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        // Calculation of seed value, Si
        let n = 2.0 * b / (v * v);
        let m = 2.0 * r / (v * v);
        let q1u = (-(n - 1.0) - (sqr(n - 1.0) + 4.0 * m).sqrt()) / 2.0;
        let su = K / (1.0 - 1.0 / q1u);
        let h1 = (b * t - 2.0 * v * t.sqrt()) * K / (K - su);
        let mut Si = su + (K - su) * f64::exp(h1);

        let k = 2.0 * r / (v * 2.0 * (1.0 - (-r * t).exp()));
        let d1 = ((Si / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let q1 = (-(n - 1.0) - (sqr(n - 1.0) + 4.0 * k).sqrt()) / 2.0;
        let mut lhs = K - Si;
        let mut rhs =
            BS::price(false, Si, K, t, r, b, v) - (1.0 - ((b - r) * t).exp() * cdf(-d1)) * Si / q1;
        let mut bi = -((b - r) * t).exp() * cdf(-d1) * (1.0 - 1.0 / q1)
            - (1.0 + ((b - r) * t).exp() * pdf(-d1) / (v * t.sqrt())) / q1;
        let epsilon = 0.000001;
        // Using the Newton Raphson algorithm, solve for Si.
        while f64::abs(lhs - rhs) / K > epsilon {
            Si = (K - rhs + bi * Si) / (1.0 + bi);
            let d1 = ((Si / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
            lhs = K - Si;
            rhs = BS::price(false, Si, K, t, r, b, v)
                - (1.0 - ((b - r) * t).exp() * cdf(-d1)) * Si / q1;
            bi = -((b - r) * t).exp() * cdf(-d1) * (1.0 - 1.0 / q1)
                - (1.0 + ((b - r) * t).exp() * cdf(-d1) / (v * t.sqrt())) / q1;
        }

        Si
    }

    #[allow(non_snake_case)]
    fn _put_price(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let Sk = BaroneAdesiWhaley::_kp(K, t, r, b, v);
        let n = 2.0 * b / (v * v);
        let k = 2.0 * r / ((v * v) * (1.0 - (-r * t).exp()));
        let d1 = ((Sk / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let q1 = (-(n - 1.0) - (sqr(n - 1.0) + 4.0 * k).sqrt()) / 2.0;
        let a1 = -(Sk / q1) * (1.0 - ((b - r) * t).exp() * cdf(-d1));

        if S > Sk {
            BS::price(false, S, K, t, r, b, v) + a1 * (S / Sk).powf(q1)
        } else {
            K - S
        }
    }

    /// The Barone-Adesi and Whaley (1987) American approximation.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if is_call {
            BaroneAdesiWhaley::_call_price(S, K, t, r, b, v)
        } else {
            BaroneAdesiWhaley::_put_price(S, K, t, r, b, v)
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
            |v| BaroneAdesiWhaley::price(is_call, S, K, t, r, b, v),
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks(is_call: bool) -> FdmWithCarry {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            BaroneAdesiWhaley::price(is_call, S, K, t, r, b, v)
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
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.087510414723946,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.51063969479627103,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8736245107976992,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.938715732901815,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.78921007059571513,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                9.4846542208284265,
            ),
        ] {
            let b = r - q;
            let actual = BaroneAdesiWhaley::price(is_call, S, K, t, r, b, v);
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
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.85926145538914866,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.10482043749123204,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.54108857322596737,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.42462427818941162,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17169091820287696,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.81309097311139666,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].delta(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
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
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018705125359730346,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018500320192460507,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042923920955395545,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.043603545223369622,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028399659594002813,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.032665941223086747,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].gamma(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
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
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.6059937555985346,
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
                -4.0673554271468033,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.2843447751640111,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.4895642722656457,
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
                ng[&is_call].theta(S, K, t, r, b, v, 1.0 / 365.0, DifferenceMethod::Central);
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
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                14.265653647129284,
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
                26.898269393325958,
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
                17.778226270531494,
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
            let numeric = ng[&is_call].vega(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
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
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                39.274484237934892,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -5.7099384558153554,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                24.580545464512713,
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
                8.0602681333376562,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -34.822340696855747,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].rho(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-10));
        }
    }
}
