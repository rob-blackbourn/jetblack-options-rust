//! Leisen Reimer binomial option pricing.

use crate::{fdm::FdmWithCarry, implied_volatility::solve_ivol, trees::Greeks};

fn sqr(x: f64) -> f64 {
    x * x
}

/// Option valuations using the Leisen Reimer method.
pub struct LeisenReimer {}

/// The following arguments are common.
///
/// * is_european (bool): Tue for European, false for American.
/// * is_call (bool): True for a call, false for a put.
/// * S (f64): The current asset price.
/// * K (f64): The option strike price
/// * t (f64): The time to maturity of the option in years.
/// * r (f64): The risk free rate.
/// * b (f64): The cost of carry of the asset.
/// * v (f64): The volatility of the asset.
/// * n (usize): The number of the steps in the tree.
/// * p (f64): The option price.
/// * max_iterations (usize): The maximum number of iterations before a price is returned.
/// * epsilon (f64): The largest acceptable error.
impl LeisenReimer {
    /// Calculate the price and some greeks using a Leisen-Reimer binomial tree.
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
        let n = if n % 2 == 0 { n + 1 } else { n };
        let z = if is_call { 1.0 } else { -1.0 };

        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        // Using Preizer-Pratt inversion method 2
        let hd1 = 0.5
            + d1.signum()
                * (0.25
                    - 0.25
                        * (-sqr(d1 / (n as f64 + 1.0 / 3.0 + 0.1 / (n + 1) as f64))
                            * (n as f64 + 1.0 / 6.0))
                            .exp())
                .sqrt();
        let hd2 = 0.5
            + d2.signum()
                * (0.25
                    - 0.25
                        * (-sqr(d2 / (n as f64 + 1.0 / 3.0 + 0.1 / (n + 1) as f64))
                            * (n as f64 + 1.0 / 6.0))
                            .exp())
                .sqrt();

        let dT = t / n as f64;
        let p = hd2;
        let u = (b * dT).exp() * hd1 / hd2;
        let d = ((b * dT).exp() - p * u) / (1.0 - p);
        let df = (-r * dT).exp();

        let mut option_value = vec![0.0; (n + 1) as usize];
        for i in 0..=n {
            option_value[i] = f64::max(
                0.0,
                z * (S * f64::powi(u, i as i32) * f64::powi(d, (n - i) as i32) - K),
            );
        }

        let mut delta = f64::NAN;
        let mut gamma = f64::NAN;
        let mut theta = f64::NAN;

        for j in (0..n).rev() {
            for i in 0..=j {
                if is_european {
                    option_value[i] = (p * option_value[i + 1] + (1.0 - p) * option_value[i]) * df;
                } else {
                    option_value[i] = f64::max(
                        z * (S * f64::powi(u, i as i32) * f64::powi(d, (j - i) as i32) - K),
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
            |v| LeisenReimer::greeks(is_european, is_call, S, K, t, r, b, v, n).price,
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks<'a>(is_european: bool, is_call: bool, n: usize) -> FdmWithCarry<'a> {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            LeisenReimer::greeks(is_european, is_call, S, K, t, r, b, v, n).price
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = (actual - expected).abs();
        diff < threshold
    }

    #[test]
    fn it_should_calc_price() {
        #[allow(non_snake_case)]
        for (is_european, is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.06954773644226,
                1e-11,
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
                0.5056518797578835,
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
                3.8694961456354164,
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
                2.913494680474701,
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
                0.7881689410025371,
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
                9.344461720850616,
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
                11.073269986822545,
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
                0.5180579183933842,
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
                3.8696227115482875,
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
                3.036361474985137,
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
                0.7881709020408987,
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
                10.098461348219312,
                1e-12,
            ),
        ] {
            let b = r - q;
            let value = LeisenReimer::greeks(is_european, is_call, S, K, t, r, b, v, 200).price;
            assert!(
                is_close_to(value, expected, threshold),
                "[{}][{}].price({}, {}, {}, {}, {}, {})",
                is_european,
                is_call,
                S,
                K,
                t,
                r,
                b,
                v
            );
        }
    }
}
