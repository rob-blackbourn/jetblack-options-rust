use libm::{exp, fabs, log, sqrt};
/// Black-Scholes-Merton options pricing formulae.
///
/// This is the "generalised" version using "cost of carry" (variable b).
///
/// The cost of carry rate (b) is:
///
/// * b == r: for non dividend paying stocks
/// * b == r - q: For dividend paying stocks where the dividend yield is q
/// * b == 0: for futures options
/// * b = r - rj: for currency options.
use std::f64::consts::PI;

use crate::distributions::inv_cdf;
use crate::{implied_volatility::solve_ivol, numeric_greeks::with_dividend_yield::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}

/// The fair value of a European option, using Black-Scholes-Merton.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to expiry of the option in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry of the asset.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The price of the options.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (b + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        S * exp((b - r) * T) * cdf(d1) - K * exp(-r * T) * cdf(d2)
    } else {
        K * exp(-r * T) * cdf(-d2) - S * exp((b - r) * T) * cdf(-d1)
    }
}

/// Calculate the volatility of an option that is implied by the price.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to expiry of the option in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry of the asset.
///     p (f64): The option price.
///     max_iterations (int, Optional): The maximum number of iterations before
///         a price is returned. Defaults to 20.
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
    b: f64,
    p: f64,
    max_iterations: Option<i32>, // = 20,
    epsilon: Option<f64>,        //=1e-8
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, b, v),
        max_iterations,
        epsilon,
    )
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
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64, v: f64| {
        price(is_call, S, K, T, r, b, v)
    })
}

/// The sensitivity of the option to a change in the asset price.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to expiry of the option in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry of the asset.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The delta.
#[allow(non_snake_case)]
pub fn delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (b + (v * v) / 2.0)) / (v * sqrt(T));

    if is_call {
        exp((b - r) * T) * cdf(d1)
    } else {
        -exp((b - r) * T) * cdf(-d1)
    }
}

/// The second derivative to the change in the asset price.
///
/// Args:
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to expiry of the option in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry of the asset.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The gamma.
#[allow(non_snake_case)]
pub fn gamma(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (b + (v * v) / 2.0)) / (v * sqrt(T));

    exp((b - r) * T) * pdf(d1) / (S * v * sqrt(T))
}

/// The theta or time decay of the value of the option.
///
/// This value is typically reported by dividing by 365 (for a one calendar day
/// movement) or 252 (for a 1 trading day movement).
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The theta.
#[allow(non_snake_case)]
pub fn theta(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        let p1 = -S * exp((b - r) * T) * pdf(d1) * v / (2.0 * sqrt(T));
        let p2 = (b - r) * S * exp((b - r) * T) * cdf(d1);
        let p3 = r * K * exp(-r * T) * cdf(d2);

        p1 - p2 - p3
    } else {
        let p1 = -S * exp((b - r) * T) * pdf(d1) * v / (2.0 * sqrt(T));
        let p2 = (b - r) * S * exp((b - r) * T) * cdf(-d1);
        let p3 = r * K * exp(-r * T) * cdf(-d2);

        p1 + p2 + p3
    }
}

/// The sensitivity of the options price or a change in the asset volatility.
///
/// This value is typically reported by dividing by 100 (for a 1% change in
/// volatility)
///
/// Args:
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to expiry of the option in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry of the asset.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The vega
#[allow(non_snake_case)]
pub fn vega(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    S * exp((b - r) * T) * pdf(d1) * sqrt(T)
}

/// The sensitivity of the option price to the risk free rate.
///
/// Useful for all options except futures options which should use
/// futures_rho.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The rho.
#[allow(non_snake_case)]
pub fn rho(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        T * K * exp(-r * T) * cdf(d2)
    } else {
        -T * K * exp(-r * T) * cdf(-d2)
    }
}

/// Sensitivity to the cost of carry.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The carry.
#[allow(non_snake_case)]
pub fn carry(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    if is_call {
        T * S * exp((b - r) * T) * cdf(d1)
    } else {
        -T * S * exp((b - r) * T) * cdf(-d1)
    }
}

/// The percentage change in the option price for a percentage change in the
/// asset price.
///
/// This is thought of as a measure of leverage, sometimes called gearing.
///
/// Also known as lambda or omega.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The elasticity.
#[allow(non_snake_case)]
pub fn elasticity(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    return delta(is_call, S, K, T, r, b, v) * S / price(is_call, S, K, T, r, b, v);
}

#[allow(non_snake_case)]
pub fn gammap(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    gamma(S, K, T, r, b, v) * S / 100.0
}

#[allow(non_snake_case)]
pub fn vegap(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    vega(S, K, T, r, b, v) * v * 10.0
}

#[allow(non_snake_case)]
pub fn forward_delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));

    if is_call {
        exp(-r * T) * cdf(d1)
    } else {
        exp(-r * T) * (cdf(d1) - 1.0)
    }
}

/// The second order derivative of the option price to a change in the asset
/// price and a change in the volatility.
///
/// Also known as DdeltaDvol.
///
/// Args:
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The vanna.
#[allow(non_snake_case)]
pub fn vanna(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    -exp((b - r) * T) * d2 / v * pdf(d1)
}

#[allow(non_snake_case)]
pub fn ddelta_dvol_dvol(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as DVannaDvol
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vanna(S, K, T, r, b, v) / v * (d1 * d2 - d1 / d2 - 1.0)
}

/// Measures the instantaneous rate of change of delta over the passage of
/// time.
///
/// Also known as DdeltaDtime.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     b (f64): The cost of carry.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The charm.
#[allow(non_snake_case)]
pub fn charm(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        -exp((b - r) * T) * (pdf(d1) * (b / (v * sqrt(T)) - d2 / (2.0 * T)) + (b - r) * cdf(d1))
    } else {
        return -exp((b - r) * T)
            * (pdf(d1) * (b / (v * sqrt(T)) - d2 / (2.0 * T)) - (b - r) * cdf(-d1));
    }
}

#[allow(non_snake_case)]
pub fn saddle_gamma(K: f64, r: f64, b: f64, v: f64) -> f64 {
    return sqrt(exp(1.0) / PI) * sqrt((2.0 * b - r) / (v * v) + 1.0) / K;
}

#[allow(non_snake_case)]
pub fn dgamma_dspot(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as Speed
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    -gamma(S, K, T, r, b, v) * (1.0 + d1 / (v * sqrt(T))) / S
}

#[allow(non_snake_case)]
pub fn dgamma_dvol(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as zomma.
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    gamma(S, K, T, r, b, v) * ((d1 * d2 - 1.0) / v)
}

#[allow(non_snake_case)]
pub fn dgamma_dtime(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    gamma(S, K, T, r, b, v) * (r - b + b * d1 / (v * sqrt(T)) + (1.0 - d1 * d2) / (2.0 * T))
}

#[allow(non_snake_case)]
pub fn dgammap_dspot(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as SpeedP.
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    return -gamma(S, K, T, r, b, v) * (d1) / (100.0 * v * sqrt(T));
}

#[allow(non_snake_case)]
pub fn dgammap_dvol(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S / 100.0 * gamma(S, K, T, r, b, v) * ((d1 * d2 - 1.0) / v)
}

#[allow(non_snake_case)]
pub fn dgammap_dtime(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    gammap(S, K, T, r, b, v) * (r - b + b * d1 / (v * sqrt(T)) + (1.0 - d1 * d2) / (2.0 * T))
}

#[allow(non_snake_case)]
pub fn dvega_dtime(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vega(S, K, T, r, b, v) * (r - b + b * d1 / (v * sqrt(T)) - (1.0 + d1 * d2) / (2.0 * T))
}

#[allow(non_snake_case)]
pub fn vomma(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as DvegaDvol
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vega(S, K, T, r, b, v) * d1 * d2 / v
}

#[allow(non_snake_case)]
pub fn dvomma_dvol(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vomma(S, K, T, r, b, v) * 1.0 / v * (d1 * d2 - d1 / d2 - d2 / d1 - 1.0)
}

#[allow(non_snake_case)]
pub fn dvegap_dvol(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as VommaP.
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    return vegap(S, K, T, r, b, v) * d1 * d2 / v;
}

#[allow(non_snake_case)]
pub fn vega_leverage(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    vega(S, K, T, r, b, v) * v / price(is_call, S, K, T, r, b, v)
}

#[allow(non_snake_case)]
pub fn variance_vega(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    S * exp((b - r) * T) * pdf(d1) * sqrt(T) / (2.0 * v)
}

#[allow(non_snake_case)]
pub fn variance_delta(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S * exp((b - r) * T) * pdf(d1) * (-d2) / (2.0 * (v * v))
}

#[allow(non_snake_case)]
pub fn variance_vomma(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S * exp((b - r) * T) * sqrt(T) / (4.0 * (v * v * v)) * pdf(d1) * (d1 * d2 - 1.0)
}

#[allow(non_snake_case)]
pub fn variance_ultima(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S * exp((b - r) * T) * sqrt(T) / (8.0 * (v * v * v * v * v))
        * pdf(d1)
        * ((d1 * d2 - 1.0) * (d1 * d2 - 3.0) - ((d1 * d1) + (d2 * d2)))
}

#[allow(non_snake_case)]
pub fn theta_driftless(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    -S * exp((b - r) * T) * pdf(d1) * v / (2.0 * sqrt(T))
}

#[allow(non_snake_case)]
pub fn futures_rho(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    -T * price(is_call, S, K, T, r, 0.0, v)
}

#[allow(non_snake_case)]
pub fn phi(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Also known as rho2.
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    if is_call {
        -T * S * exp((b - r) * T) * cdf(d1)
    } else {
        T * S * exp((b - r) * T) * cdf(-d1)
    }
}

#[allow(non_snake_case)]
pub fn dzeta_dvol(is_call: bool, S: f64, K: f64, T: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        -pdf(d2) * d1 / v
    } else {
        pdf(d2) * d1 / v
    }
}

#[allow(non_snake_case)]
pub fn dzeta_dtime(is_call: bool, S: f64, K: f64, T: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        return pdf(d2) * (b / (v * sqrt(T)) - d1 / (2.0 * T));
    } else {
        return -pdf(d2) * (b / (v * sqrt(T)) - d1 / (2.0 * T));
    }
}

#[allow(non_snake_case)]
pub fn break_even_probability(
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
) -> f64 {
    // Risk neutral break even probability.
    if is_call {
        let K = K + price(true, S, K, T, r, b, v) * exp(r * T);
        let d2 = (log(S / K) + (b - (v * v) / 2.0) * T) / (v * sqrt(T));
        return cdf(d2);
    } else {
        let K = K - price(false, S, K, T, r, b, v) * exp(r * T);
        let d2 = (log(S / K) + (b - (v * v) / 2.0) * T) / (v * sqrt(T));
        return cdf(-d2);
    }
}

#[allow(non_snake_case)]
pub fn strike_delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d2 = (log(S / K) + (b - (v * v) / 2.0) * T) / (v * sqrt(T));
    if is_call {
        -exp(-r * T) * cdf(d2)
    } else {
        exp(-r * T) * cdf(-d2)
    }
}

#[allow(non_snake_case)]
pub fn risk_neutral_density(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let d2 = (log(S / K) + (b - (v * v) / 2.0) * T) / (v * sqrt(T));
    exp(-r * T) * pdf(d2) / (K * v * sqrt(T))
}

#[allow(non_snake_case)]
pub fn gamma_from_delta(S: f64, T: f64, r: f64, b: f64, v: f64, delta_: f64) -> f64 {
    exp((b - r) * T) * pdf(inv_cdf(exp((r - b) * T) * fabs(delta_))) / (S * v * sqrt(T))
}

#[allow(non_snake_case)]
pub fn gammap_from_delta(S: f64, T: f64, r: f64, b: f64, v: f64, delta_: f64) -> f64 {
    S / 100.0 * gamma_from_delta(S, T, r, b, v, delta_)
}

#[allow(non_snake_case)]
pub fn vega_from_delta(S: f64, T: f64, r: f64, b: f64, delta_: f64) -> f64 {
    S * exp((b - r) * T) * sqrt(T) * pdf(inv_cdf(exp((r - b) * T) * fabs(delta_)))
}

#[allow(non_snake_case)]
pub fn vegap_from_delta(S: f64, T: f64, r: f64, b: f64, v: f64, delta_: f64) -> f64 {
    v / 10.0 * vega_from_delta(S, T, r, b, delta_)
}

#[allow(non_snake_case)]
pub fn strike_from_delta(
    is_call: bool,
    S: f64,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
    delta_: f64,
) -> f64 {
    if is_call {
        S * exp(-inv_cdf(delta_ * exp((r - b) * T)) * v * sqrt(T) + (b + v * v / 2.0) * T)
    } else {
        S * exp(inv_cdf(-delta_ * exp((r - b) * T)) * v * sqrt(T) + (b + v * v / 2.0) * T)
    }
}

#[allow(non_snake_case)]
pub fn in_the_money_prob_from_delta(
    is_call: bool,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
    delta_: f64,
) -> f64 {
    if is_call {
        cdf(inv_cdf(delta_ / exp((b - r) * T)) - v * sqrt(T))
    } else {
        cdf(inv_cdf(-delta_ / exp((b - r) * T)) + v * sqrt(T))
    }
}

#[allow(non_snake_case)]
pub fn strike_from_in_the_money_prob(
    is_call: bool,
    S: f64,
    v: f64,
    T: f64,
    b: f64,
    in_the_money_prob: f64,
) -> f64 {
    if is_call {
        return S * exp(-inv_cdf(in_the_money_prob) * v * sqrt(T) + (b - (v * v) / 2.0) * T);
    } else {
        return S * exp(inv_cdf(in_the_money_prob) * v * sqrt(T) + (b - (v * v) / 2.0) * T);
    }
}

#[allow(non_snake_case)]
pub fn rnd_from_in_the_money_prob(K: f64, T: f64, r: f64, v: f64, in_the_money_prob: f64) -> f64 {
    exp(-r * T) * pdf(inv_cdf(in_the_money_prob)) / (K * v * sqrt(T))
}

#[allow(non_snake_case)]
pub fn delta_from_in_the_money_prob(
    is_call: bool,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
    in_the_money_prob: f64,
) -> f64 {
    if is_call {
        return cdf(inv_cdf(in_the_money_prob * exp((b - r) * T)) - v * sqrt(T));
    } else {
        return -cdf(inv_cdf(in_the_money_prob * exp((b - r) * T)) + v * sqrt(T));
    }
}

/// What asset price that gives maximum DdeltaDvol
///
/// is_lower == True gives lower asset level that gives max DdeltaDvol
/// is_lower == False gives upper asset level that gives max DdeltaDvol
#[allow(non_snake_case)]
pub fn max_ddelta_dvol_asset(is_lower: bool, K: f64, T: f64, b: f64, v: f64) -> f64 {
    if is_lower {
        return K * exp(-b * T - v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0);
    } else {
        return K * exp(-b * T + v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0);
    }
}

/// What strike price that gives maximum DdeltaDvol
///
/// is_lower == True gives lower strike level that gives max DdeltaDvol
/// is_lower == False gives upper strike level that gives max DdeltaDvol
#[allow(non_snake_case)]
pub fn max_ddelta_dvol_strike(is_lower: bool, S: f64, T: f64, b: f64, v: f64) -> f64 {
    if is_lower {
        S * exp(b * T - v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0)
    } else {
        S * exp(b * T + v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0)
    }
}

/// What strike price that gives maximum gamma and vega
#[allow(non_snake_case)]
pub fn max_gamma_vega_at_X(S: f64, b: f64, T: f64, v: f64) -> f64 {
    S * exp((b + (v * v) / 2.0) * T)
}

/// What asset price that gives maximum gamma
#[allow(non_snake_case)]
pub fn max_gamma_at_S(x: f64, b: f64, T: f64, v: f64) -> f64 {
    x * exp((-b - 3.0 * (v * v) / 2.0) * T)
}

/// What asset price that gives maximum vega
#[allow(non_snake_case)]
pub fn max_vega_at_S(K: f64, b: f64, T: f64, v: f64) -> f64 {
    K * exp((-b + (v * v) / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn in_the_money_probability(is_call: bool, S: f64, K: f64, T: f64, b: f64, v: f64) -> f64 {
    let d2 = (log(S / K) + (b - (v * v) / 2.0) * T) / (v * sqrt(T));

    if is_call { cdf(d2) } else { cdf(-d2) }
}

#[allow(non_snake_case)]
pub fn delta_mirror_strike(S: f64, T: f64, b: f64, v: f64) -> f64 {
    return S * exp((b + (v * v) / 2.0) * T);
}

#[allow(non_snake_case)]
pub fn probability_mirror_strike(S: f64, T: f64, b: f64, v: f64) -> f64 {
    S * exp((b - (v * v) / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn delta_mirror_call_put_strike(S: f64, K: f64, T: f64, b: f64, v: f64) -> f64 {
    (S * S) / K * exp((2.0 * b + (v * v)) * T)
}

#[allow(non_snake_case)]
pub fn profit_loss_std(
    is_cash: bool,
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    b: f64,
    v: f64,
    n_hedges: i32,
) -> f64 {
    if is_cash {
        sqrt(PI / 4.0) * vega(S, K, T, r, b, v) * v / sqrt(n_hedges.into())
    } else {
        sqrt(PI / 4.0) * vega(S, K, T, r, b, v) * v
            / sqrt(n_hedges.into())
            / price(is_call, S, K, T, r, b, v)
    }
}
