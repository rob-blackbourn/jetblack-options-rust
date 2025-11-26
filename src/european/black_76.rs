//! # Black (1976) Options on futures/forwards
//!
//!
//! * The discounted futures price $ F $,
//! * Strike price $ K $,
//! * Risk-free rate $ r $,
//! * Annual dividend yield $ q $,
//! * Time to maturity $ \tau = T - t $
//! * Volatility $ \sigma $.
//!
//! Most of the formula use one or both of the following terms.
//!
//! $$
//! d_1 = \frac{\ln(F/K) + (\sigma^2/2)T}{\sigma\sqrt{T}}
//! $$
//!
//! $$
//! d_2 = \frac{\ln(F/K) - (\sigma^2/2)T}{\sigma\sqrt{T}} = d_1 - \sigma\sqrt{T}
//! $$
//!
//! $$
//! \varphi(x) &= \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2}
//! $$
//!
//! $$
//! \Phi(x) &= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{1}{2} y^2} \,dy = 1 - \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-\frac{1}{2} y^2} \,dy
//! $$

use libm::{exp, log, sqrt};

use crate::{implied_volatility::solve_ivol, numeric_greeks::without_carry::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}

/// ## price
///
/// Fair value of a futures/forward using Black 76.
///
/// For a call:
///
/// $$
/// C = e^{-r \tau}[F\Phi(d_1) - K\Phi(d_2)]
/// $$
///
/// For a put:
///
/// $$
/// P = e^{-r \tau} [K\Phi(-d_2) -  F\Phi(-d_1)]
/// $$
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * F (f64): The price of the future.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The asset volatility.
///
/// ### Returns
///
/// f64: The option price.
#[allow(non_snake_case)]
pub fn price(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        exp(-r * T) * (F * cdf(d1) - K * cdf(d2))
    } else {
        exp(-r * T) * (K * cdf(-d2) - F * cdf(-d1))
    }
}

/// ## ivol
///
/// Calculate the volatility of a Black 76 option that is implied by the price.
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * F (f64): The current asset price.
/// * K (f64): The option strike price
/// * T (f64): The time to maturity of the option in years.
/// * r (f64): The risk free rate.
/// * p (f64): The option price.
/// * max_iterations (int, Optional): The maximum number of iterations before
///     a price is returned. Defaults to 20.
/// * epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.
///
/// ### Returns
///
/// f64: The implied volatility.
#[allow(non_snake_case)]
pub fn ivol(
    is_call: bool,
    F: f64,
    K: f64,
    T: f64,
    r: f64,
    p: f64,
    max_iterations: Option<i32>,
    epsilon: Option<f64>,
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, F, K, T, r, v),
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
/// * NumericGreeks: A class which can generate Greeks using finite difference
///     methods.
pub fn make_numeric_greeks(is_call: bool) -> NumericGreeks {
    // Normalize the price function to match that required by the finite
    // difference methods.
    #[allow(non_snake_case)]
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64| price(is_call, S, K, T, r, b))
}

/// ## delta
///
/// The sensitivity of the option to a change in the asset price
/// using Black 76.
///
/// For the call.
///
/// $$
/// \frac{\partial C}{\partial S} = e^{-r \tau} \Phi(d_1)
/// $$
///
/// For the put.
///
/// $$
/// \frac{\partial P}{\partial S} = -e^{-r \tau} \Phi(-d_1)
/// $$
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * F (f64): The current futures price.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The volatility.
///
/// ### Returns
///
/// f64: The delta.
#[allow(non_snake_case)]
pub fn delta(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));
    if is_call {
        return exp(-r * T) * cdf(d1);
    } else {
        return -exp(-r * T) * cdf(-d1);
    }
}

/// ## gamma
///
/// The second derivative to the change in asset price using Black 76.
///
/// The gamma for both calls and puts.
///
/// $$
/// \frac{\partial^2 V}{\partial S^2} = e^{-r \tau} \frac{\varphi(d_1)}{F\sigma\sqrt{\tau}} = K e^{-r \tau} \frac{\varphi(d_2)}{F^2\sigma\sqrt{\tau}}
/// $$
///
/// ### Arguments
///
/// * F (f64): The current futures price.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The volatility.
///
/// ### Returns
///
/// f64: The gamma.
#[allow(non_snake_case)]
pub fn gamma(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));

    exp(-r * T) * pdf(d1) / (F * v * sqrt(T))
}

/// ## theta
///
/// The change in the value of the option with respect to time to expiry
/// using Black 76.
///
/// For the call.
///
/// $$
/// \frac{\partial C}{\partial T} - \frac{F e^{-r \tau} \varphi(d_1) \sigma}{2 \sqrt{\tau}} - rKe^{-r \tau}\Phi(d_2) + rFe^{-r \tau}\Phi(d_1)
/// $$
///
/// For the put.
///
/// $$
/// \frac{\partial P}{\partial T} - \frac{F e^{-r \tau} \varphi(d_1) \sigma}{2 \sqrt{\tau}} + rKe^{-r \tau}\Phi(-d_2) - rFe^{-r \tau}\Phi(-d_1)
/// $$
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * F (f64): The current futures price.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The volatility.
///
/// ### Returns
///
/// f64: The theta.
#[allow(non_snake_case)]
pub fn theta(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        -F * exp(-r * T) * pdf(d1) * v / (2.0 * sqrt(T)) + r * F * exp(-r * T) * cdf(d1)
            - r * K * exp(-r * T) * cdf(d2)
    } else {
        -F * exp(-r * T) * pdf(d1) * v / (2.0 * sqrt(T)) - r * F * exp(-r * T) * cdf(-d1)
            + r * K * exp(-r * T) * cdf(-d2)
    }
}

/// ## vega
///
/// The sensitivity of the options price or a change in the asset volatility
/// using Black 76.
///
/// For both calls and puts.
///
/// $$
/// \frac{\partial V}{\partial \sigma} = F e^{-r \tau} \varphi(d_1) \sqrt{\tau} = K e^{-r \tau} \varphi(d_2) \sqrt{\tau}
/// $$
///
/// ### Arguments
///
/// * F (f64): The current futures price.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The volatility.
///
/// ### Returns
///
/// f64: The vega.
#[allow(non_snake_case)]
pub fn vega(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));

    F * exp(-r * T) * pdf(d1) * sqrt(T)
}

/// ## rho
///
/// The sensitivity of the option price to a change in the risk free rate
/// using Black 76.
///
/// For a call:
///
/// $$
/// \frac{\partial C}{\partial r} = -\tau e^{-r \tau}[F\Phi(d_1) - K\Phi(d_2)]
/// $$
///
/// For a put:
///
/// $$
/// \frac{\partial P}{\partial r} = -\tau e^{-r \tau} [K\Phi(-d_2) -  F\Phi(-d_1)]
/// $$
///
/// ### Arguments
///
/// * is_call (bool): True for a call, false for a put.
/// * F (f64): The price of the future.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The asset volatility.
///
/// ### Returns
///
/// f64: The rho.
#[allow(non_snake_case)]
pub fn rho(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        -T * exp(-r * T) * (F * cdf(d1) - K * cdf(d2))
    } else {
        -T * exp(-r * T) * (K * cdf(-d2) - F * cdf(-d1))
    }
}

/// ## vanna
///
/// The sensitivity of the option value to the underlying
/// asset price and the volatility.
///
/// For both calls and puts.
///
/// $$
/// \frac{\partial^2 V}{\partial F \partial \sigma} = -e^{-r \tau} \varphi(d_1) \frac{d_2}{\sigma} \, = \frac{\mathcal{V}}{F}\left[1 - \frac{d_1}{\sigma\sqrt{\tau}} \right]
/// $$
///
/// ### Arguments
///
/// * F (f64): The price of the future.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The asset volatility.
///
/// ### Returns
///
/// f64: The vanna.
#[allow(non_snake_case)]
pub fn vanna(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    -exp(-r * T) * pdf(d1) * d2 / v
}

/// ## vomma
///
/// The second order sensitivity to volatility.
///
/// For both puts and calls.
///
/// $$
/// \frac{\partial^2 V}{\partial \sigma^2} = F e^{-r \tau} \varphi(d_1) \sqrt{\tau} \frac{d_1 d_2}{\sigma} = \mathcal{V}  \frac{d_1 d_2}{\sigma}
/// $$
///
/// ### Arguments
///
/// * F (f64): The price of the future.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * v (f64): The asset volatility.
///
/// ### Returns
///
/// f64: The vomma
#[allow(non_snake_case)]
pub fn vomma(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    F * exp(-r * T) * pdf(d1) * sqrt(T) * d1 * d2 / v
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, f64};

    use libm::fabs;

    use crate::numeric_greeks::DifferenceMethod;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        fabs(actual - expected) < threshold
    }

    #[test]
    fn it_should_calc_price() {
        #[allow(non_snake_case)]
        for (is_call, F, K, T, r, v, expected) in [
            (
                true,
                110.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                10.143390791460092,
            ),
            (
                false,
                110.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                0.63109654645295865,
            ),
            (
                true,
                100.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                3.3531192847248605,
            ),
            (
                false,
                100.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                3.3531192847248534,
            ),
            (
                true,
                100.0,
                110.0,
                6.0 / 12.0,
                0.1,
                0.125,
                0.63109654645296376,
            ),
            (
                false,
                100.0,
                110.0,
                6.0 / 12.0,
                0.1,
                0.125,
                10.143390791460105,
            ),
        ] {
            let actual = price(is_call, F, K, T, r, v);
            assert!(is_close_to(actual, expected, f64::EPSILON));
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
                10.143390791460092,
                0.125,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.6310965464529535,
                0.125,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                3.3531192847248605,
                0.125,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                3.3531192847248534,
                0.125,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.6310965464529654,
                0.125,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                10.143390791460092,
                0.125,
            ),
        ] {
            let actual = ivol(is_call, F, K, T, r, p, None, None);
            assert!(is_close_to(actual, expected, 1e-9));
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, F, K, r, T, v, expected, numeric_threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.82678604419568191,
                1e-11,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.12444338030503208,
                1e-10,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.4923803086739813,
                1e-10,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.45884911582673277,
                1e-10,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.14319868380006498,
                1e-10,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.80803074070064895,
                1e-9,
            ),
        ] {
            let analytic = delta(is_call, F, K, T, r, v);
            assert!(is_close_to(analytic, expected, f64::EPSILON));

            let numerical =
                ng[&is_call].delta(F, K, T, r, v, Some(0.0001), Some(DifferenceMethod::Central));
            assert!(
                is_close_to(numerical, analytic, numeric_threshold),
                "[{}].delta({}, {}, {}, {}, {}) -> {} <diff={:e}>",
                is_call,
                F,
                K,
                T,
                r,
                v,
                numerical,
                analytic - numerical
            );
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, F, K, r, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.020787293603316447,
                1e-8,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.020787293603316447,
                1e-8,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.042891991459829734,
                1e-8,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.042891991459829734,
                1e-8,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.025152625260012922,
                1e-9,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                0.025152625260012922,
                1e-9,
            ),
        ] {
            let analytic = gamma(F, K, T, r, v);
            assert!(is_close_to(analytic, expected, f64::EPSILON));

            let numeric = ng[&is_call].gamma(F, K, T, r, v, Some(0.01), None);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].gamma({}, {}, {}, {}, {}) -> {} (diff={:e})",
                is_call,
                F,
                K,
                T,
                r,
                v,
                numeric,
                analytic - numeric
            );
        }
    }

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, F, K, r, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.95070976929249973,
                1e-8,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -1.901939193793212,
                1e-8,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -3.0156249043267125,
                1e-8,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -3.0156249043267138,
                1e-8,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -1.9019391937932129,
                1e-9,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.95070976929249973,
                1e-9,
            ),
        ] {
            let analytic = theta(is_call, F, K, T, r, v);
            assert!(is_close_to(analytic, expected, f64::EPSILON));

            let numerical = ng[&is_call].theta(
                F,
                K,
                T,
                r,
                v,
                Some(1.0 / 365.0 / 24.0 / 60.0),
                Some(DifferenceMethod::Central),
            );
            assert!(
                is_close_to(numerical, analytic, threshold),
                "[{}].theta({}, {}, {}, {}, {}) -> {} (diff={:e})",
                is_call,
                F,
                K,
                T,
                r,
                v,
                numerical,
                analytic - numerical
            );
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, F, K, r, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                15.720390787508064,
                1e-7,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                15.720390787508064,
                1e-7,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                26.807494662393587,
                1e-9,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                26.807494662393587,
                1e-9,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                15.720390787508077,
                1e-7,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                15.720390787508077,
                1e-7,
            ),
        ] {
            let analytic = vega(F, K, T, r, v);
            assert!(is_close_to(analytic, expected, f64::EPSILON));

            let numerical = ng[&is_call].vega(
                F,
                K,
                T,
                r,
                v,
                Some(0.00001),
                Some(DifferenceMethod::Central),
            );
            assert!(
                is_close_to(numerical, analytic, threshold),
                "[{}].vega({}, {}, {}, {}, {}) -> {} (diff={:e})",
                is_call,
                F,
                K,
                T,
                r,
                v,
                numerical,
                analytic - numerical
            );
        }
    }

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, F, K, r, T, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -5.0716953957300461,
                1e-10,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.31554827322647933,
                1e-11,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -1.6765596423624303,
                1e-10,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -1.6765596423624267,
                1e-11,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -0.31554827322648188,
                1e-11,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.125,
                -5.0716953957300523,
                1e-10,
            ),
        ] {
            let analytic = rho(is_call, F, K, T, r, v);
            assert!(is_close_to(analytic, expected, f64::EPSILON));

            let numerical = ng[&is_call].rho(
                F,
                K,
                T,
                r,
                v,
                Some(0.00001),
                Some(DifferenceMethod::Central),
            );
            assert!(
                is_close_to(numerical, analytic, threshold),
                "[{}].rho({}, {}, {}, {}, {}) -> {} (diff={:e})",
                is_call,
                F,
                K,
                T,
                r,
                v,
                numerical,
                analytic - numerical
            );
        }
    }

    #[test]
    fn it_should_calc_vanna() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, _q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.6720354862986977,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.6720354862986977,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.13403747331196794,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.13403747331196794,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                1.996442942803649,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                1.996442942803649,
            ),
        ] {
            let analytic = vanna(S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vanna(S, K, T, r, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_vomma() {
        let ng = HashMap::from([
            (true, make_numeric_greeks(true)),
            (false, make_numeric_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, _q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                145.98618448189166,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                145.98618448189166,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.41886710409989986,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.41886710409989986,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                145.98618448189166,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                145.98618448189166,
            ),
        ] {
            let analytic = vomma(S, K, T, r, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vomma(S, K, T, r, v, None);
            assert!(is_close_to(numeric, analytic, 1e-2));
        }
    }
}
