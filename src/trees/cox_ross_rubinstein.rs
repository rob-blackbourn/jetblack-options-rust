//! Cox, Ross & Rubinstein

use libm::{exp, fmax, pow, sqrt};

use crate::{fdm::FdmWithCarry, implied_volatility::solve_ivol, trees::Greeks};

/// Option valuation implementations using the Cox, Ross & Rubinstein
/// binomial tree.
pub struct CoxRossRubinstein {}

/// An implementation of a Cox, Ross & Rubinstein binomial tree model.
///
/// The following arguments are common:
///
/// * is_european (bool): True for European, false for American.
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
impl CoxRossRubinstein {
    /// A Cox-Ross-Rubinstein binomial tree option pricer returning the price and some greeks.
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

        let dT = t / (n as f64);
        let u = exp(v * sqrt(dT));
        let d = 1.0 / u;
        let a = exp(b * dT);
        let p = (a - d) / (u - d);
        let df = exp(-r * dT);

        let mut option_value = vec![0.0; n + 1];
        for i in 0..option_value.len() {
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
                gamma = ((option_value[2] - option_value[1]) / (S * (u * u) - S)
                    - (option_value[1] - option_value[0]) / (S - S * (d * d)))
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
            |v| CoxRossRubinstein::greeks(is_european, is_call, S, K, t, r, b, v, n).price,
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks(is_european: bool, is_call: bool, n: usize) -> FdmWithCarry {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            CoxRossRubinstein::greeks(is_european, is_call, S, K, t, r, b, v, n).price
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
                11.070810696746085,
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
                0.5069148400613138,
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
                3.865263891560955,
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
                2.909262426399531,
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
                0.7893184818731922,
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
                9.34561126171898,
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
                11.074659452363559,
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
                0.5190977256236272,
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
                3.8653919092508406,
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
                3.034209281414657,
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
                0.7893202834043941,
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
                10.100588559015748,
                1e-12,
            ),
        ] {
            let b = r - q;
            let value =
                CoxRossRubinstein::greeks(is_european, is_call, S, K, t, r, b, v, 200).price;
            assert!(is_close_to(value, expected, threshold));
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (
                true,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(true, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(true, false, 100)),
                ]),
            ),
            (
                false,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(false, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(false, false, 100)),
                ]),
            ),
        ]);

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
                0.8606397006611921,
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
                -0.10014973849180042,
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
                0.5403040031344286,
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
                -0.42048543601840294,
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
                0.166152892001592,
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
                -0.7946365471511285,
                1e-12,
            ),
        ] {
            let b = r - q;
            let numeric =
                ng[&is_european][&is_call].delta(S, K, t, r, b, v, 1e-2, DifferenceMethod::Central);
            assert!(is_close_to(numeric, expected, threshold));
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (
                true,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(true, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(true, false, 100)),
                ]),
            ),
            (
                false,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(false, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(false, false, 100)),
                ]),
            ),
        ]);

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
                -5.3290705182007514e-11,
                1e-10,
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
                1.6653345369377348e-11,
                1e-11,
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
                7.552788665350363,
                1e-10,
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
                7.5527886653592454,
                1e-10,
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
                0.0,
                1e-11,
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
                1.0658141036401503e-10,
                1e-10,
            ),
        ] {
            let b = r - q;
            let actual =
                ng[&is_european][&is_call].gamma(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            let diff = fabs(expected - actual);
            assert!(
                diff < threshold,
                "[{}][{}].gamma({}, {}, {}, {}, {}, {})",
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

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([
            (
                true,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(true, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(true, false, 100)),
                ]),
            ),
            (
                false,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(false, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(false, false, 100)),
                ]),
            ),
        ]);

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
                -2.4761276405278343,
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
                -1.4187804088872178,
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
                -4.032411057869996,
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
                -2.2064322687333013,
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
                0.34573941621434123,
                1e-12,
            ),
        ] {
            let b = r - q;
            let numeric = ng[&is_european][&is_call].theta(
                S,
                K,
                t,
                r,
                b,
                v,
                1.0 / 365.0,
                DifferenceMethod::Central,
            );
            assert!(is_close_to(numeric, expected, threshold));
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([
            (
                true,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(true, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(true, false, 100)),
                ]),
            ),
            (
                false,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(false, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(false, false, 100)),
                ]),
            ),
        ]);

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
                13.589147303977533,
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
                13.589147303972009,
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
                26.703128947528132,
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
                26.70312894752369,
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
                17.34480813834477,
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
                17.34480813833894,
                1e-11,
            ),
        ] {
            let b = r - q;
            let actual =
                ng[&is_european][&is_call].vega(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            let diff = fabs(expected - actual);
            assert!(
                diff < threshold,
                "[{}][{}].vega({}, {}, {}, {}, {}, {})",
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

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([
            (
                true,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(true, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(true, false, 100)),
                ]),
            ),
            (
                false,
                HashMap::from([
                    (true, CoxRossRubinstein::fdm_greeks(false, true, 100)),
                    (false, CoxRossRubinstein::fdm_greeks(false, false, 100)),
                ]),
            ),
        ]);

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
                41.58039938613545,
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
                -5.981073820944649,
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
                25.084660240106025,
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
                -22.476812966961024,
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
                8.165255681767846,
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
                -44.152364845996495,
                1e-11,
            ),
        ] {
            let b = r - q;
            let actual =
                ng[&is_european][&is_call].rho(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            let diff = fabs(expected - actual);
            assert!(
                diff < threshold,
                "[{}][{}].rho({}, {}, {}, {}, {}, {})",
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
