//! # Option pricing functions implementing the Barone, Adesi and Whaley (1987)
//!
//! American approximation.

use libm::{exp, fabs, log, pow, sqrt};

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

/// ## _kc
///
/// Newton Raphson algorithm to solve for the critical commodity price for a call.
///
/// ### Arguments
///
/// * K (f64): The strike.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * b (f64): The asset growth.
/// * v (f64): The volatility.
///
/// ### Returns
///
/// f64: The price.
#[allow(non_snake_case)]
fn _kc(K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Calculate the seed value Si
    let n = 2.0 * b / (v * v);
    let m = 2.0 * r / (v * v);
    let q2u = (-(n - 1.0) + sqrt(sqr(n - 1.0) + 4.0 * m)) / 2.0;
    let su = K / (1.0 - 1.0 / q2u);
    let h2 = -(b * T + 2.0 * v * sqrt(T)) * K / (su - K);
    let mut Si = K + (su - K) * (1.0 - exp(h2));

    let k = 2.0 * r / ((v * v) * (1.0 - exp(-r * T)));
    let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let q2 = (-(n - 1.0) + sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
    let mut lhs = Si - K;
    let mut rhs = bs_price(true, Si, K, T, r, b, v) + (1.0 - exp((b - r) * T) * cdf(d1)) * Si / q2;
    let mut bi = exp((b - r) * T) * cdf(d1) * (1.0 - 1.0 / q2)
        + (1.0 - exp((b - r) * T) * cdf(d1) / (v * sqrt(T))) / q2;
    let epsilon = 0.000001;
    // Using the Newton Raphson algorithm solve for Si
    while fabs(lhs - rhs) / K > epsilon {
        Si = (K + rhs - bi * Si) / (1.0 - bi);
        let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
        lhs = Si - K;
        rhs = bs_price(true, Si, K, T, r, b, v) + (1.0 - exp((b - r) * T) * cdf(d1)) * Si / q2;
        bi = exp((b - r) * T) * cdf(d1) * (1.0 - 1.0 / q2)
            + (1.0 - exp((b - r) * T) * pdf(d1) / (v * sqrt(T))) / q2;
    }

    Si
}

#[allow(non_snake_case)]
fn _call_price(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    if b >= r {
        return bs_price(true, S, K, T, r, b, v);
    }

    let Sk = _kc(K, T, r, b, v);
    let n = 2.0 * b / (v * v);
    let k = 2.0 * r / ((v * v) * (1.0 - exp(-r * T)));
    let d1 = (log(Sk / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let q2 = (-(n - 1.0) + sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
    let a2 = (Sk / q2) * (1.0 - exp((b - r) * T) * cdf(d1));
    if S < Sk {
        bs_price(true, S, K, T, r, b, v) + a2 * pow(S / Sk, q2)
    } else {
        S - K
    }
}

/// ## _kp
///
/// Newton Raphson algorithm to solve for the critical commodity price for a put.
///
/// ### Arguments
///
/// * K (f64): The strike.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * b (f64): The asset growth.
/// * v (f64): The volatility.
///
/// ### Returns
///
/// f64: The price.
#[allow(non_snake_case)]
fn _kp(K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    // Calculation of seed value, Si
    let n = 2.0 * b / (v * v);
    let m = 2.0 * r / (v * v);
    let q1u = (-(n - 1.0) - sqrt(sqr(n - 1.0) + 4.0 * m)) / 2.0;
    let su = K / (1.0 - 1.0 / q1u);
    let h1 = (b * T - 2.0 * v * sqrt(T)) * K / (K - su);
    let mut Si = su + (K - su) * exp(h1);

    let k = 2.0 * r / (v * 2.0 * (1.0 - exp(-r * T)));
    let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let q1 = (-(n - 1.0) - sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
    let mut lhs = K - Si;
    let mut rhs =
        bs_price(false, Si, K, T, r, b, v) - (1.0 - exp((b - r) * T) * cdf(-d1)) * Si / q1;
    let mut bi = -exp((b - r) * T) * cdf(-d1) * (1.0 - 1.0 / q1)
        - (1.0 + exp((b - r) * T) * pdf(-d1) / (v * sqrt(T))) / q1;
    let epsilon = 0.000001;
    // Using the Newton Raphson algorithm, solve for Si.
    while fabs(lhs - rhs) / K > epsilon {
        Si = (K - rhs + bi * Si) / (1.0 + bi);
        let d1 = (log(Si / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
        lhs = K - Si;
        rhs = bs_price(false, Si, K, T, r, b, v) - (1.0 - exp((b - r) * T) * cdf(-d1)) * Si / q1;
        bi = -exp((b - r) * T) * cdf(-d1) * (1.0 - 1.0 / q1)
            - (1.0 + exp((b - r) * T) * cdf(-d1) / (v * sqrt(T))) / q1;
    }

    Si
}

#[allow(non_snake_case)]
fn _put_price(S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    let Sk = _kp(K, T, r, b, v);
    let n = 2.0 * b / (v * v);
    let k = 2.0 * r / ((v * v) * (1.0 - exp(-r * T)));
    let d1 = (log(Sk / K) + (b + (v * v) / 2.0) * T) / (v * sqrt(T));
    let q1 = (-(n - 1.0) - sqrt(sqr(n - 1.0) + 4.0 * k)) / 2.0;
    let a1 = -(Sk / q1) * (1.0 - exp((b - r) * T) * cdf(-d1));

    if S > Sk {
        bs_price(false, S, K, T, r, b, v) + a1 * pow(S / Sk, q1)
    } else {
        K - S
    }
}

/// ## price
///
/// The Barone-Adesi and Whaley (1987) American approximation.
///
/// ### Arguments
///
/// * is_call (bool): true for a call, false for a put.
/// * S (f64): The asset price.
/// * K (f64): The strike price.
/// * T (f64): The time to expiry in years.
/// * r (f64): The risk free rate.
/// * b (f64): The cost of carry.
/// * v (f64): The asset volatility.
///
/// ### Returns
///
/// f64: The price of the option.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, b: f64, v: f64) -> f64 {
    if is_call {
        _call_price(S, K, T, r, b, v)
    } else {
        _put_price(S, K, T, r, b, v)
    }
}

/// ## ivol
///
/// Calculate the volatility of an option that is implied by the price.
///
/// ### Arguments
///
/// * is_call (bool): true for a call, false for a put.
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
