use crate::distributions::{cdf, pdf};
use crate::european::GeneralizedBlackScholes as BS;
use crate::fdm::FdmWithCarry;
use crate::implied_volatility::solve_ivol;

fn sqr(x: f64) -> f64 {
    x * x
}

/// # Option pricing functions using Barone, Adesi and Whaley (1987)
///
/// American approximation.
pub struct BaroneAdesiWhaley {}

/// The following arguments are common.
///
/// * K (f64): The strike.
/// * t (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * b (f64): The asset growth.
/// * v (f64): The volatility.
/// * max_iterations (usize): The maximum number of iterations before a price is returned.
/// * epsilon (f64): The largest acceptable error.
impl BaroneAdesiWhaley {
    /// Solve for the critical commodity price for a call using the Newton Raphson algorithm.
    #[allow(non_snake_case)]
    fn kc(K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
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
    fn call_price(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if b >= r {
            return BS::price(true, S, K, t, r, b, v);
        }

        let Sk = BaroneAdesiWhaley::kc(K, t, r, b, v);
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

    /// Solve for the critical commodity price for a put using the Newton Raphson algorithm.
    #[allow(non_snake_case)]
    fn kp(K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
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
    fn put_price(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let Sk = BaroneAdesiWhaley::kp(K, t, r, b, v);
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

    /// The fair value of an American option using Barone, Adesi & Whaley.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        if is_call {
            BaroneAdesiWhaley::call_price(S, K, t, r, b, v)
        } else {
            BaroneAdesiWhaley::put_price(S, K, t, r, b, v)
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
    pub fn fdm_greeks<'a>(is_call: bool) -> FdmWithCarry<'a> {
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
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.087510414723946,
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
                0.51063969479627103,
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
                3.8736245107976992,
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
                2.938715732901815,
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
                0.78921007059571513,
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
                9.4846542208284265,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = BaroneAdesiWhaley::price(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(actual, expected, threshold),
                "price({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e}",
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
            )
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
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
                0.85926145538914866,
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
                -0.10482043749123204,
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
                0.54108857322596737,
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
                -0.42462427818941162,
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
                0.17169091820287696,
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
                -0.81309097311139666,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].delta(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
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
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
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
                0.018705125359730346,
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
                0.018500320192460507,
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
                0.042923920955395545,
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
                0.043603545223369622,
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
                0.028399659594002813,
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
                0.032665941223086747,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].gamma(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(
                is_close_to(actual, expected, threshold),
                "[{}].gamma({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
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
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
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
                -2.605998373930074,
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
                -1.4788337133328777,
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
                -4.0673414540277175,
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
                -2.2843311017558543,
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
                -2.489572897110087,
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
                0.11715070857878374,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].theta(
                S,
                K,
                t,
                r,
                b,
                v,
                1.0 / 365.0 / 60.0,
                DifferenceMethod::Central,
            );
            assert!(
                is_close_to(actual, expected, threshold),
                "[{}].theta({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
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
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
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
                14.265653647129284,
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
                13.9833977312675,
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
                26.898269393325958,
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
                26.853425641593763,
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
                17.778226270531494,
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
                16.181880739890353,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual = ng[&is_call].vega(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(actual, expected, 1e-11),
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
            (true, BaroneAdesiWhaley::fdm_greeks(true)),
            (false, BaroneAdesiWhaley::fdm_greeks(false)),
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
                39.274484237934892,
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
                -5.7099384558153554,
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
                24.580545464512713,
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
                -20.964770823997945,
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
                8.0602681333376562,
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
                -34.822340696855747,
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
