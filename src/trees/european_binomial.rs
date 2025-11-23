//! # Optional valuation with a European binomial implementation.

use libm::{exp, log, pow, sqrt};

use crate::{
    distributions::binomial_coefficient::comb, implied_volatility::solve_ivol,
    numeric_greeks::with_carry::NumericGreeks,
};

/// ## price
///
/// A European binomial option pricing tree.
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * S (f64): The current asset price.
/// * K (f64): The option strike price
/// * T (f64): The time to maturity of the option in years.
/// * r (f64): The risk free rate.
/// * b (f64): The cost of carry of the asset.
/// * v (f64): The volatility of the asset.
/// * n (usize): The number of the steps in the tree.
///
/// ### Returns
///
/// f64: The price of the option.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64, n: u64) -> f64 {
    let dt = T / n as f64;
    let u = exp(v * sqrt(dt));
    let d = 1.0 / u;
    let a = exp(b * dt);
    let p = (a - d) / (u - d);
    let A = (log(K / (S * pow(d, n as f64))) / log(u / d)) as u64 + 1;

    let mut sum = 0.0;
    if is_call {
        for j in A..=n {
            sum += comb(n as u64, j as u64)
                * pow(p, j as f64)
                * pow(1.0 - p, (n - j) as f64)
                * (S * pow(u, j as f64) * pow(d, (n - j) as f64) - K);
        }
    } else {
        for j in 0..A {
            sum += comb(n, j)
                * pow(p, j as f64)
                * pow(1.0 - p, (n - j) as f64)
                * (K - S * pow(u, j as f64) * pow(d, (n - j) as f64));
        }
    }

    exp(-r * T) * sum
}

/// ## ivol
///
/// Calculate the volatility of an option that is implied by the price.
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * S (f64): The current asset price.
/// * K (f64): The option strike price
/// * T (f64): The time to expiry of the option in years.
/// * r (f64): The risk free rate.
/// * b (f64): The cost of carry of the asset.
/// * p (f64): The option price.
/// * n (u64): The number of the steps in the tree.
/// * max_iterations (u64, Optional): The maximum number of iterations before
///       a price is returned. Defaults to 20.
/// * epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.
///
/// ### Returns
///
/// f64: The implied volatility.
#[allow(non_snake_case)]
pub fn ivol(
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    b: f64,
    p: f64,
    n: u64,
    max_iterations: Option<i32>, // = 20,
    epsilon: Option<f64>,        // =1e-8
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, b, v, n),
        max_iterations,
        epsilon,
    )
}

/// ## make_numeric_greeks
///
/// Make a class to generate greeks numerically using finite difference methods.
///
/// ### Arguments
///
/// * is_call (bool): If true the options is a call;  otherwise it is a put.
///
/// ### Returns
///
/// NumericGreeks: A class which can generate Greeks using finite difference
///  methods.
pub fn make_numeric_greeks(is_call: bool, n: u64) -> NumericGreeks {
    // Normalize the price function to match that required by the finite
    // difference methods.
    #[allow(non_snake_case)]
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64, v: f64| {
        price(is_call, S, K, T, r, b, v, n)
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use libm::fabs;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = fabs(actual - expected);
        diff < threshold
    }

    #[test]
    fn it_should_calc_price() {
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
                11.070810696746085,
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
                0.5069148400613138,
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
                3.865263891560955,
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
                2.909262426399531,
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
                0.7893184818731922,
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
                9.34561126171898,
                1e-12,
            ),
        ] {
            let b = r - q;
            let value = price(is_call, S, K, T, r, b, v, 200);
            assert!(is_close_to(value, expected, threshold));
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true, 100)),
            (false, make_numeric_greeks(false, 100)),
        ]);

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
                0.8606397006611921,
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
                -0.10014973849180042,
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
                0.5403040031344286,
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
                -0.42048543601840294,
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
                0.166152892001592,
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
                -0.7946365471511285,
                1e-12,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].delta(S, K, T, r, b, v, None, None);
            assert!(is_close_to(numeric, expected, threshold));
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true, 100)),
            (false, make_numeric_greeks(false, 100)),
        ]);

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
                -5.3290705182007514e-11,
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
                1.1102230246251565e-11,
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
                7.552788665377008,
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
                7.552788665390331,
                1e-10,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                4.440892098500626e-12,
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
                -1.7763568394002505e-11,
                1e-10,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].gamma(S, K, T, r, b, v, None, None);
            assert!(
                is_close_to(numeric, expected, threshold),
                "[{}].gamma({}, {}, {}, {}, {}, {})",
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
        let ng = HashMap::from([
            (true, make_numeric_greeks(true, 100)),
            (false, make_numeric_greeks(false, 100)),
        ]);

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
                -2.4761276405391808,
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
                -1.4187804088877445,
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
                -4.032411057874453,
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
                -2.206432268736381,
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
                -2.431468809334608,
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
                0.3457394162046157,
                1e-12,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_call].theta(S, K, T, r, b, v, None, None);
            assert!(is_close_to(numeric, expected, threshold));
        }
    }
}
