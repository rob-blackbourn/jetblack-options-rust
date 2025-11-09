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
    return solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, v),
        max_iterations,
        epsilon,
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
