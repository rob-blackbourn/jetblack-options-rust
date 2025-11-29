//! # Jarrow-Rudd binomial pricing tree
//!
//! Option valuations using a Jarrow-Rudd binomial pricing tree.
//!
//! The following arguments are common
//!
//! * is_european (bool): Tue for European, false for American.
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The current asset price.
//! * K (f64): The option strike price
//! * T (f64): The time to maturity of the option in years.
//! * r (f64): The risk free rate.
//! * b (f64): The cost of carry of the asset.
//! * v (f64): The volatility of the asset.
//! * n (usize): The number of the steps in the tree.
//! * p (f64): The option price.
//! * max_iterations (usize): The maximum number of iterations before a price is returned.
//! * epsilon (f64): The largest acceptable error.

use core::f64;

use libm::{exp, fmax, pow, sqrt};

use crate::{fdm::FdmWithCarry, implied_volatility::solve_ivol};

/// The greeks returned by the model.
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
}

/// Calculate the price and some greeks using a Jarrow-Rudd binomial option pricing tree.
#[allow(non_snake_case)]
pub fn greeks(
    is_european: bool,
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
    n: usize,
) -> Greeks {
    let z = if is_call { 1.0 } else { -1.0 };

    let dT = T / n as f64;
    let u = exp((b - (v * v) / 2.0) * dT + v * sqrt(dT));
    let d = exp((b - (v * v) / 2.0) * dT - v * sqrt(dT));
    let p = 0.5;
    let df = exp(-r * dT);

    let mut option_value = vec![0.0; n + 1];
    for i in 0..=n {
        option_value[i] = fmax(0.0, z * (S * pow(u, i as f64) * pow(d, (n - i) as f64) - K));
    }

    let mut delta = f64::NAN;
    let mut gamma = f64::NAN;
    let mut theta = f64::NAN;

    for j in (0..n).rev() {
        for i in 0..=j {
            if is_european {
                option_value[i] = (p * option_value[i + 1] + (1.0 - p) * option_value[i]) * df;
            } else {
                option_value[i] = fmax(
                    z * (S * pow(u, i as f64) * pow(d, (j - i) as f64) - K),
                    (p * option_value[i + 1] + (1.0 - p) * option_value[i]) * df,
                );
            }
        }

        if j == 2 {
            gamma = ((option_value[2] - option_value[1]) / (S * (u * u) - S * u * d)
                - (option_value[1] - option_value[0]) / (S * u * d - S * (d * d)))
                / (0.5 * (S * (u * u) - S * (d * d)));
            theta = option_value[1];
        }

        if j == 1 {
            delta = (option_value[1] - option_value[0]) / (S * u - S * d);
        }
    }

    theta = (theta - option_value[0]) / (2.0 * dT) / 365.0;

    return Greeks {
        price: option_value[0],
        delta,
        gamma,
        theta,
    };
}

/// Calculate the price using a Jarrow-Rudd binomial option pricing tree.
#[allow(non_snake_case)]
pub fn price(
    is_european: bool,
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
    n: usize,
) -> f64 {
    greeks(is_european, is_call, S, K, T, r, b, v, n).price
}

/// Calculate the volatility of an option that is implied by the price.
#[allow(non_snake_case)]
pub fn ivol(
    is_european: bool,
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    b: f64,
    p: f64,
    n: usize,
    max_iterations: usize,
    epsilon: f64,
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_european, is_call, S, K, T, r, b, v, n),
        max_iterations,
        epsilon,
    )
}

/// Return a struct to calculate greeks numerically using finite difference methods.
pub fn fdm_greeks(is_european: bool, is_call: bool, n: usize) -> FdmWithCarry {
    #[allow(non_snake_case)]
    FdmWithCarry::new(move |S: f64, K: f64, T: f64, r: f64, b: f64, v: f64| {
        price(is_european, is_call, S, K, T, r, b, v, n)
    })
}

#[cfg(test)]
mod tests {
    use libm::fabs;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = fabs(actual - expected);
        diff < threshold
    }

    #[test]
    fn it_should_calc_price() {
        #[allow(non_snake_case)]
        for (is_european, is_call, S, K, r, q, T, v, expected, threshold) in [
            (
                true,
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.068143516623877,
                1e-12,
            ),
            (
                true,
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5042503476664845,
                1e-12,
            ),
            (
                true,
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.873668938664346,
                1e-12,
            ),
            (
                true,
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.917669916891453,
                1e-12,
            ),
            (
                true,
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.7869051773732514,
                1e-12,
            ),
            (
                true,
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                9.343200400607568,
                1e-12,
            ),
            (
                false,
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.071997737972442,
                1e-12,
            ),
            (
                false,
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5164002396952895,
                1e-12,
            ),
            (
                false,
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8737974667835564,
                1e-12,
            ),
            (
                false,
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.040445420072334,
                1e-12,
            ),
            (
                false,
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.786906995272888,
                1e-12,
            ),
            (
                false,
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                10.099966374860383,
                1e-12,
            ),
        ] {
            let b = r - q;
            let value = price(is_european, is_call, S, K, T, r, b, v, 200);
            assert!(is_close_to(value, expected, threshold));
        }
    }
}
