//! # Option valuation functions implementing the Bjerksund and Stensland (2002)
//!
//! American approximation
//!
//! The following arguments are common:
//!
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The current asset price.
//! * K (f64): The option strike price
//! * T (f64): The time to maturity of the option in years.
//! * r (f64): The risk free rate.
//! * b (f64): The cost of carry of the asset.
//! * v (f64): The volatility of the asset.
//! * p (f64): The option price.
//! * max_iterations (usize): The maximum number of iterations before a price is returned.
//! * epsilon (f64): The largest acceptable error.

use libm::{exp, fmax, log, pow, sqrt};

use crate::distributions::cbnd::cbnd;
use crate::european::generalized_black_scholes::price as bs_price;
use crate::{implied_volatility::solve_ivol, fdm::with_carry::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}

fn sqr(x: f64) -> f64 {
    x * x
}

#[allow(non_snake_case)]
fn _phi(S: f64, T: f64, gamma_: f64, h: f64, i: f64, r: f64, b: f64, v: f64) -> f64 {
    let lambda_ = (-r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1.0) * (v * v)) * T;
    let d = -(log(S / h) + (b + (gamma_ - 0.5) * (v * v)) * T) / (v * sqrt(T));
    let kappa = 2.0 * b / (v * v) + 2.0 * gamma_ - 1.0;
    exp(lambda_)
        * pow(S, gamma_)
        * (cdf(d) - pow(i / S, kappa) * cdf(d - 2.0 * log(i / S) / (v * sqrt(T))))
}

#[allow(non_snake_case)]
fn _ksi(
    S: f64,
    T2: f64,
    gamma_: f64,
    h: f64,
    I2: f64,
    I1: f64,
    t1: f64,
    r: f64,
    b: f64,
    v: f64,
) -> f64 {
    let e1 = (log(S / I1) + (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));
    let e2 = (log((I2 * I2) / (S * I1)) + (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));
    let e3 = (log(S / I1) - (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));
    let e4 = (log(I2 * I2 / (S * I1)) - (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));

    let f1 = (log(S / h) + (b + (gamma_ - 0.5) * (v * v)) * T2) / (v * sqrt(T2));
    let f2 = (log((I2 * I2) / (S * h)) + (b + (gamma_ - 0.5) * (v * v)) * T2) / (v * sqrt(T2));
    let f3 = (log((I1 * I1) / (S * h)) + (b + (gamma_ - 0.5) * (v * v)) * T2) / (v * sqrt(T2));
    let f4 = (log(S * (I1 * I1) / (h * (I2 * I2))) + (b + (gamma_ - 0.5) * (v * v)) * T2)
        / (v * sqrt(T2));

    let rho = sqrt(t1 / T2);
    let lambda_ = -r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1.0) * (v * v);
    let kappa = 2.0 * b / (v * v) + (2.0 * gamma_ - 1.0);

    exp(lambda_ * T2)
        * pow(S, gamma_)
        * (cbnd(-e1, -f1, rho)
            - pow(I2 / S, kappa) * cbnd(-e2, -f2, rho)
            - pow(I1 / S, kappa) * cbnd(-e3, -f3, -rho)
            + pow(I1 / I2, kappa) * cbnd(-e4, -f4, -rho))
}

#[allow(non_snake_case)]
fn _call_price(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let t1 = 1.0 / 2.0 * (sqrt(5.0) - 1.0) * T;

    if b >= r {
        // Use Black-Scholes as it is never optimal to exercise before maturity.
        return bs_price(true, S, K, T, r, b, v);
    }

    let beta = (1.0 / 2.0 - b / (v * v)) + sqrt(sqr(b / (v * v) - 1.0 / 2.0) + 2.0 * r / (v * v));
    let b_infinity = beta / (beta - 1.0) * K;
    let b0 = fmax(K, r / (r - b) * K);

    let ht1 = -(b * t1 + 2.0 * v * sqrt(t1)) * (K * K) / ((b_infinity - b0) * b0);
    let ht2 = -(b * T + 2.0 * v * sqrt(T)) * (K * K) / ((b_infinity - b0) * b0);
    let I1 = b0 + (b_infinity - b0) * (1.0 - exp(ht1));
    let I2 = b0 + (b_infinity - b0) * (1.0 - exp(ht2));
    let alfa1 = (I1 - K) * pow(I1, -beta);
    let alfa2 = (I2 - K) * pow(I2, -beta);

    if S >= I2 {
        S - K
    } else {
        return alfa2 * pow(S, beta) - alfa2 * _phi(S, t1, beta, I2, I2, r, b, v)
            + _phi(S, t1, 1.0, I2, I2, r, b, v)
            - _phi(S, t1, 1.0, I1, I2, r, b, v)
            - K * _phi(S, t1, 0.0, I2, I2, r, b, v)
            + K * _phi(S, t1, 0.0, I1, I2, r, b, v)
            + alfa1 * _phi(S, t1, beta, I1, I2, r, b, v)
            - alfa1 * _ksi(S, T, beta, I1, I2, I1, t1, r, b, v)
            + _ksi(S, T, 1.0, I1, I2, I1, t1, r, b, v)
            - _ksi(S, T, 1.0, K, I2, I1, t1, r, b, v)
            - K * _ksi(S, T, 0.0, I1, I2, I1, t1, r, b, v)
            + K * _ksi(S, T, 0.0, K, I2, I1, t1, r, b, v);
    }
}

/// The Bjerksund and Stensland (2002) American approximation.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    if is_call {
        _call_price(S, K, T, r, b, v)
    } else {
        // Use the Bjerksund and Stensland put-call transformation
        _call_price(K, S, T, r - b, -b, v)
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
        |v| price(is_call, S, K, T, r, b, v),
        max_iterations,
        epsilon,
    )
}

/// Return a struct to calculate greeks numerically using finite difference methods.
pub fn fdm_greeks(is_call: bool) -> NumericGreeks {
    #[allow(non_snake_case)]
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64, v: f64| {
        price(is_call, S, K, T, r, b, v)
    })
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
                11.071608152504766,
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
            ),
        ] {
            let b = r - q;
            let actual = price(is_call, S, K, T, r, b, v);

            assert!(is_close_to(actual, expected, 1e-12));
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([(true, fdm_greeks(true)), (false, fdm_greeks(false))]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.857387269854204,
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
                -0.10603562406572564,
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
                0.5404729148441589,
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
                -0.44432538850074366,
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
                0.17153035196955102,
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
                -0.9106399407253107,
                1e-12,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].delta(S, K, T, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, threshold));
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([(true, fdm_greeks(true)), (false, fdm_greeks(false))]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018553563609913226,
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
                0.01894102808819298,
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
                0.028376564245036207,
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
            let numeric = ng[&is_call].gamma(S, K, T, r, b, v, 0.01, DifferenceMethod::Central);
            let diff = fabs(expected - numeric);
            assert!(
                diff < threshold,
                "[{}].gammma({}, {}, {}, {}, {}, {})",
                is_call,
                S,
                K,
                T,
                r,
                b,
                v
            );
        }
    }

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([(true, fdm_greeks(true)), (false, fdm_greeks(false))]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.530993738452243,
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
                -4.040856526609158,
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
            let numeric =
                ng[&is_call].theta(S, K, T, r, b, v, 1.0 / 365.0, DifferenceMethod::Central);
            assert!(
                fabs(expected - numeric) < threshold,
                "[{}]theta({}, {}, {}, {}, {}, {})",
                is_call,
                S,
                K,
                T,
                r,
                b,
                v
            );
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([(true, fdm_greeks(true)), (false, fdm_greeks(false))]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.978277991237853,
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
            let numeric = ng[&is_call].vega(S, K, T, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, expected, threshold),
                "[{}].vega({}, {}, {}, {}, {}, {})",
                is_call,
                S,
                K,
                T,
                r,
                b,
                v
            );
        }
    }

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([(true, fdm_greeks(true)), (false, fdm_greeks(false))]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, v, expected, threshold) in [
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
                -5.692145861011966,
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
                25.06962040217786,
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
            let numeric = ng[&is_call].rho(S, K, T, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, expected, threshold),
                "[{}].rho({}, {}, {}, {}, {}, {})",
                is_call,
                S,
                K,
                T,
                r,
                b,
                v
            );
        }
    }
}
