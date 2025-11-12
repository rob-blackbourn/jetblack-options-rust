/// Black-Scholes 1973.
///
/// The original Black-Scholes option formula for an option on a non-dividend
/// paying stock option.
///
/// $$
/// d_1 = \frac{1}{\sigma\sqrt{T - t}}\left[\ln\left(\frac{S_t}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)(T - t)\right]
/// $$
///
/// $$
/// d_2 = d_1 - \sigma\sqrt{T - t}
/// $$
use libm::{exp, log, sqrt};

use crate::{implied_volatility::solve_ivol, numeric_greeks::without_carry::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}

/// Black-Scholes for a non-dividend paying stock.
///
/// $$
/// C(S_t, t) = N(d_1)S_t - N(d_2)Ke^{-r(T - t)}
/// $$
///
/// $$
/// P(S_t, t) = N(-d_2) Ke^{-r(T - t)} - N(-d_1) S_t
/// $$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The price of the option.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        S * cdf(d1) - K * exp(-r * T) * cdf(d2)
    } else {
        K * exp(-r * T) * cdf(-d2) - S * cdf(-d1)
    }
}

/// Calculate the volatility of a Black-Scholes 73 option that is implied by
/// the price.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     p (f64): The option price.
///     max_iterations (int, Optional): The maximum number of iterations before
///         a price is returned. Defaults to 35.
///     epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.
///
/// Returns:
///     f64: The implied volatility.
#[allow(non_snake_case)]
pub fn ivol(
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    p: f64,
    max_iterations: Option<i32>, // = 35,
    epsilon: Option<f64>,        // =1e-8
) -> f64 {
    let max_iterations = max_iterations.unwrap_or(35);
    let epsilon = epsilon.unwrap_or(1e-8);
    return solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, v),
        Some(max_iterations),
        Some(epsilon),
    );
}

/// Make a class to generate greeks numerically using finite difference methods.
///
/// Args:
///     is_call (bool): If true the options is a call;  otherwise it is a put.
///
/// Returns:
///     NumericGreeks: A class which can generate Greeks using finite difference
///         methods.
pub fn make_numeric_greeks(is_call: bool) -> NumericGreeks {
    // Normalize the price function to match that required by the finite
    // difference methods.
    #[allow(non_snake_case)]
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64| price(is_call, S, K, T, r, b))
}

#[allow(non_snake_case)]
pub fn delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    if is_call { cdf(d1) } else { -cdf(-d1) }
}

/// Calculates option gamma
#[allow(non_snake_case)]
pub fn gamma(S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    pdf(d1) / (S * v * sqrt(T))
}

/// Calculates option theta
#[allow(non_snake_case)]
pub fn theta(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        -((S * pdf(d1) * v) / (2.0 * sqrt(T))) - r * K * exp(-r * T) * cdf(d2)
    } else {
        -((S * pdf(d1) * v) / (2.0 * sqrt(T))) + r * K * exp(-r * T) * cdf(-d2)
    }
}

#[allow(non_snake_case)]
pub fn vega(S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    S * sqrt(T) * pdf(d1)
}

#[allow(non_snake_case)]
pub fn rho(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        K * T * exp(-r * T) * cdf(d2)
    } else {
        -K * T * exp(-r * T) * cdf(-d2)
    }
}

// Also known as DdeltaDvol.
#[allow(non_snake_case)]
pub fn vanna(S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    -d2 * pdf(d1) / v
}

/// Also known as DdeltaDtime
#[allow(non_snake_case)]
pub fn charm(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        -pdf(d1) * (r / (v * sqrt(T)) - d2 / (2.0 * T))
    } else {
        -pdf(d1) * (r / (v * sqrt(T)) - d2 / (2.0 * T))
    }
}

/// Also known as DvegaDvol
#[allow(non_snake_case)]
pub fn vomma(S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vega(S, K, T, r, v) * d1 * d2 / v
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
        for (is_call, S, K, r, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                15.066208620179964,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.18915107025137257,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                6.413154785988965,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                1.536097236060364,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                1.7525027662779316,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                6.387739461356475,
            ),
        ] {
            let actual = price(is_call, S, K, T, r, v);
            assert!(is_close_to(actual, expected, 1e-12))
        }
    }

    #[test]
    fn it_should_calc_ivol() {
        #[allow(non_snake_case)]
        for (is_call, F, K, r, T, p, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                15.066208620179964,
                0.125,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.18915107025137257,
                0.125,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                6.413154785988965,
                0.125,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                1.536097236060364,
                0.125,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                1.7525027662779316,
                0.125,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                6.387739461356475,
                0.125,
            ),
        ] {
            let actual = ivol(is_call, F, K, T, r, p, None, None);
            assert!(is_close_to(actual, expected, 1e-6))
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.9543127330810848,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.045687266918915226,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.7290292160988521,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.2709707839011479,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.3197378469845452,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.6802621530154548,
            ),
        ] {
            let analytic = delta(is_call, S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].delta(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-6))
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.009868587816478383,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.009868587816478383,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.037475415422440525,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.037475415422440525,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.040445177937028856,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.040445177937028856,
            ),
        ] {
            let analytic = gamma(S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].gamma(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-6));
        }
    }

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -9.923709143900409,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.41141489889326743,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -9.576743512267791,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.06444926726065026,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -6.181907719548038,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                4.281615949959816,
            ),
        ] {
            let analytic = theta(is_call, S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].theta(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                7.463119536211778,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                7.463119536211778,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                23.42213463902533,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                23.42213463902533,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                25.278236210643035,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                25.278236210643035,
            ),
        ] {
            let analytic = vega(S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vega(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-3));
        }
    }

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                44.954096009369676,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -2.6073752156660236,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                33.244883411948123,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -14.316587813087578,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                15.110640966088296,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -37.20697738145098,
            ),
        ] {
            let analytic = rho(is_call, S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].rho(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_vanna() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
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
                -1.2280022470048304,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.2280022470048304,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.3819059437024943,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.3819059437024943,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                1.5924538086889684,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                1.5924538086889684,
            ),
        ] {
            let analytic = vanna(S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vanna(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_charm() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
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
                0.044945814894341574,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.044945814894341574,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.20201591126159346,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.20201591126159346,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.6035085054564097,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.6035085054564097,
            ),
        ] {
            let analytic = charm(is_call, S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].charm(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-5));
        }
    }

    #[test]
    fn it_should_calc_vomma() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
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
                161.24953775897959,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                161.24953775897959,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                59.59469382217008,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                59.59469382217008,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                52.74707656927028,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                52.74707656927028,
            ),
        ] {
            let analytic = vomma(S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vomma(S, K, T, r, v, None);
            assert!(is_close_to(numeric, analytic, 1e-2));
        }
    }
}
