//! # Option pricing functions implementing the Bjerksund and Stensland (1993)
//!
//! American approximation.

use libm::{exp, fmax, log, pow, sqrt};

use crate::european::generalised_black_scholes::price as bs_price;
use crate::{implied_volatility::solve_ivol, numeric_greeks::with_carry::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}

fn sqr(x: f64) -> f64 {
    x * x
}

#[allow(non_snake_case)]
fn _phi(S: f64, T: f64, gamma_: f64, h: f64, i: f64, r: f64, b: f64, v: f64) -> f64 {
    let lambda_ = (-r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1.0) * (v * v)) * T;
    let d = -(log(S / h) + (b + (gamma_ - 0.5) * (v * v)) * T) / (v * sqrt(T));
    let kappa = 2.0 * b / (v * v) + 2.0 * gamma_ - 1.0;
    exp(lambda_)
        * pow(S, gamma_)
        * (cdf(d) - pow(i / S, kappa) * cdf(d - 2.0 * log(i / S) / (v * sqrt(T))))
}

#[allow(non_snake_case)]
fn _call_price(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    if b >= r {
        // We can use Black-Scholes as it is never optimal to exercise before
        // maturity.
        return bs_price(true, S, K, T, r, b, v);
    }

    let beta = (1.0 / 2.0 - b / (v * v)) + sqrt(sqr(b / (v * v) - 1.0 / 2.0) + 2.0 * r / (v * v));
    let b_infinity = beta / (beta - 1.0) * K;
    let b0 = fmax(K, r / (r - b) * K);
    let ht = -(b * T + 2.0 * v * sqrt(T)) * b0 / (b_infinity - b0);
    let i = b0 + (b_infinity - b0) * (1.0 - exp(ht));
    let alpha = (i - K) * pow(i, -beta);
    if S >= i {
        S - K
    } else {
        alpha * pow(S, beta) - alpha * _phi(S, T, beta, i, i, r, b, v)
            + _phi(S, T, 1.0, i, i, r, b, v)
            - _phi(S, T, 1.0, K, i, r, b, v)
            - K * _phi(S, T, 0.0, i, i, r, b, v)
            + K * _phi(S, T, 0.0, K, i, r, b, v)
    }
}

/// ## price
///
/// The Bjerksund and Stensland (1993) American approximation.
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
///
/// ### Returns
///
/// f64: The price of the option.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    if is_call {
        _call_price(S, K, T, r, b, v)
    } else {
        // Use the Bjerksund and Stensland put-call transformation
        _call_price(K, S, T, r - b, -b, v)
    }
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
/// * max_iterations (int, Optional): The maximum number of iterations before
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
    max_iterations: Option<i32>, // = 20,
    epsilon: Option<f64>,        // =1e-8
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, b, v),
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
///     methods.
pub fn make_numeric_greeks(is_call: bool) -> NumericGreeks {
    // Normalize the price function to match that required by the finite
    // difference methods.
    #[allow(non_snake_case)]
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64, v: f64| {
        price(is_call, S, K, T, r, b, v)
    })
}
