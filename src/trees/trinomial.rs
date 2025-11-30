//! # Option valuations using a trinomial tree.
//!
//! The following arguments are common.
//!
//! * is_european (bool): True for European, false for American.
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The current asset price.
//! * K (f64): The option strike price
//! * t (f64): The time to maturity of the option in years.
//! * r (f64): The risk free rate.
//! * b (f64): The cost of carry of the asset.
//! * v (f64): The volatility of the asset.
//! * n (usize): The number of the steps in the tree.
//! * p (f64): The option price.
//! * max_iterations (usize): The maximum number of iterations before a price is returned.
//! * epsilon (f64): The largest acceptable error.

use libm::{exp, fmax, sqrt};
use std::cmp::max;

use crate::{fdm::FdmWithCarry, implied_volatility::solve_ivol, trees::Greeks};

fn sqr(x: f64) -> f64 {
    x * x
}

fn ipow(x: f64, p: i32) -> f64 {
    let mut sum = 1.0;
    for _ in 0..p {
        sum *= x;
    }
    sum
}

pub struct Trinomial {}

impl Trinomial {
    /// A trinomial tree options pricer returning the price and some greeks.
    #[allow(non_snake_case)]
    pub fn greeks(
        is_european: bool,
        is_call: bool,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        b: f64,
        v: f64,
        n: usize,
    ) -> Greeks {
        let z = if is_call { 1.0 } else { -1.0 };

        let dT = t / n as f64;
        let u = exp(v * sqrt(2.0 * dT));
        let d = exp(-v * sqrt(2.0 * dT));
        let pu = sqr((exp(b * dT / 2.0) - exp(-v * sqrt(dT / 2.0)))
            / (exp(v * sqrt(dT / 2.0)) - exp(-v * sqrt(dT / 2.0))));
        let pd = sqr((exp(v * sqrt(dT / 2.0)) - exp(b * dT / 2.0))
            / (exp(v * sqrt(dT / 2.0)) - exp(-v * sqrt(dT / 2.0))));
        let pm = 1.0 - pu - pd;
        let Df = exp(-r * dT);

        let mut option_value = vec![0.0; 1 + 2 * n];
        for i in 0..option_value.len() {
            let I = i as i32;
            let N = n as i32;
            option_value[i] = fmax(
                0.0,
                z * (S * ipow(u, max(I - N, 0)) * ipow(d, max(N - I, 0)) - K),
            );
        }

        let mut delta = f64::NAN;
        let mut gamma = f64::NAN;
        let mut theta = f64::NAN;

        for j in (0..n).rev() {
            for i in 0..=(j * 2) {
                option_value[i] =
                    (pu * option_value[i + 2] + pm * option_value[i + 1] + pd * option_value[i])
                        * Df;

                if is_european {
                    let I = i as i32;
                    let J = j as i32;
                    option_value[i] = fmax(
                        z * (S * ipow(u, max(I - J, 0)) * ipow(d, max(J - I, 0)) - K),
                        option_value[i],
                    );
                }
            }

            if j == 1 {
                delta = (option_value[2] - option_value[0]) / (S * u - S * d);
                gamma = ((option_value[2] - option_value[1]) / (S * u - S)
                    - (option_value[1] - option_value[0]) / (S - S * d))
                    / (0.5 * (S * u - S * d));
                theta = option_value[1];
            }
        }

        theta = (theta - option_value[0]) / dT / 365.0;

        return Greeks {
            price: option_value[0],
            delta,
            gamma,
            theta,
        };
    }

    /// Calculate the volatility of an option that is implied by the price.
    #[allow(non_snake_case)]
    pub fn ivol(
        is_european: bool,
        is_call: bool,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        b: f64,
        p: f64,
        n: usize,
        max_iterations: usize,
        epsilon: f64,
    ) -> f64 {
        solve_ivol(
            p,
            |v| Trinomial::greeks(is_european, is_call, S, K, t, r, b, v, n).price,
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks(is_european: bool, is_call: bool, n: usize) -> FdmWithCarry {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            Trinomial::greeks(is_european, is_call, S, K, t, r, b, v, n).price
        })
    }
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
        for (is_european, is_call, S, K, r, q, t, v, price, threshold) in [
            (
                true,
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.073487046391277,
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
                0.5177978858542516,
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
                3.8675119431883433,
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
                3.0348490093800793,
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
                0.7880335120168521,
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
                10.099148293480416,
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
                11.069621888515023,
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
                0.5057260318285591,
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
                3.867381427247627,
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
                2.911379962084615,
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
                0.7880316003855702,
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
                9.344324380229715,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = Trinomial::greeks(is_european, is_call, S, K, t, r, b, v, 200);
            assert!(
                is_close_to(actual.price, price, threshold),
                "price({}, {}, {}, {}, {}, {}, {}, {}, {})",
                is_european,
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                200
            );
        }
    }
}
