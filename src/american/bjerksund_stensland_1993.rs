//! # Option pricing functions implementing the Bjerksund and Stensland (1993)
//!
//! American approximation.
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

use crate::european::GeneralizedBlackScholes as BS;
use crate::{fdm::FdmWithCarry, implied_volatility::solve_ivol};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}

fn sqr(x: f64) -> f64 {
    x * x
}

pub struct BjerksundStensland1993 {}

impl BjerksundStensland1993 {
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
    fn _call_price(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if b >= r {
            // We can use Black-Scholes as it is never optimal to exercise before
            // maturity.
            return BS::price(true, S, K, t, r, b, v);
        }

        let beta =
            (1.0 / 2.0 - b / (v * v)) + (sqr(b / (v * v) - 1.0 / 2.0) + 2.0 * r / (v * v)).sqrt();
        let b_infinity = beta / (beta - 1.0) * K;
        let b0 = f64::max(K, r / (r - b) * K);
        let ht = -(b * t + 2.0 * v * t.sqrt()) * b0 / (b_infinity - b0);
        let i = b0 + (b_infinity - b0) * (1.0 - f64::exp(ht));
        let alpha = (i - K) * f64::powf(i, -beta);
        if S >= i {
            S - K
        } else {
            alpha * f64::powf(S, beta)
                - alpha * BjerksundStensland1993::_phi(S, t, beta, i, i, r, b, v)
                + BjerksundStensland1993::_phi(S, t, 1.0, i, i, r, b, v)
                - BjerksundStensland1993::_phi(S, t, 1.0, K, i, r, b, v)
                - K * BjerksundStensland1993::_phi(S, t, 0.0, i, i, r, b, v)
                + K * BjerksundStensland1993::_phi(S, t, 0.0, K, i, r, b, v)
        }
    }

    /// The Bjerksund and Stensland (1993) American approximation.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if is_call {
            BjerksundStensland1993::_call_price(S, K, t, r, b, v)
        } else {
            // Use the Bjerksund and Stensland put-call transformation
            BjerksundStensland1993::_call_price(K, S, t, r - b, -b, v)
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
            |v| BjerksundStensland1993::price(is_call, S, K, t, r, b, v),
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks(is_call: bool) -> FdmWithCarry {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            BjerksundStensland1993::price(is_call, S, K, t, r, b, v)
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
                11.070181515952816,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5100272464024442,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8695089570482253,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.999829098372267,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.7881686046834773,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                10.08179174018241,
            ),
        ] {
            let b = r - q;
            let actual = BjerksundStensland1993::price(is_call, S, K, t, r, b, v);
            assert!(is_close_to(actual, expected, 1e-12));
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, BjerksundStensland1993::fdm_greeks(true)),
            (false, BjerksundStensland1993::fdm_greeks(false)),
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
                0.8569705161634467,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.10549822644847495,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5404561878748382,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.4430050293766641,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17153014578425996,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.9142855581178999,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].delta(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-11));
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, BjerksundStensland1993::fdm_greeks(true)),
            (false, BjerksundStensland1993::fdm_greeks(false)),
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
                0.018450076488818468,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018827352619155135,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.04283403384874873,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.04811970086393558,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.02837645780573439,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.04419040397607432,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].gamma(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, 1e-9));
        }
    }
}
