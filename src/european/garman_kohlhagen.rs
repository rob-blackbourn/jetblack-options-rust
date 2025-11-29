//! # Garman and Kohlhagen (1983) Currency options.
//!
//! The value of a call option.
//!
//! $$
//! c = S_0e^{-r_f T}\mathcal{N}(d_1) - Ke^{-r_d T}\mathcal{N}(d_2)
//! $$
//!
//! The value of a put option.
//!
//! $$
//! p = Ke^{-r_d T}\mathcal{N}(-d_2) - S_0e^{-r_f T}\mathcal{N}(-d_1)
//! $$
//!
//! where:
//!
//! $$
//! d_1 = \frac{\ln(S_0/K) + (r_d - r_f + \sigma^2/2)T}{\sigma\sqrt{T}}
//! $$
//!
//! and
//!
//! $$
//! d_2 = d_1 - \sigma\sqrt{T}
//! $$
//!
//! * $S_0$ is the current spot rate
//! * $K$ is the strike price
//! * $\mathcal{N}(x)$ is the cumulative normal distribution function
//! * $r_d$ is domestic risk free [[simple interest]] rate
//! * $r_f$ is foreign risk free simple interest rate
//! * $T$ is the time to maturity (calculated according to the appropriate day count convention)
//! * $\sigma$ is the volatility of the FX rate.
//!
//! Command arguments are:
//!
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The asset price.
//! * K (f64): The strike price.
//! * T (f64): The time to expiry in years.
//! * r (f64): The risk free rate of the base currency.
//! * rf (f64): The risk free rate of the quote currency.
//! * v (f64): The asset volatility.
//! * max_iterations (int, Optional): The maximum number of iterations before a price is returned. Defaults to 20.
//! * epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.

use libm::{exp, log, sqrt};

use crate::{fdm::with_dividend_yield::NumericGreeks, implied_volatility::solve_ivol};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}

/// The fair value of a currency option.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, rf: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (r - rf + (v * v) / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        S * exp(-rf * T) * cdf(d1) - K * exp(-r * T) * cdf(d2)
    } else {
        K * exp(-r * T) * cdf(-d2) - S * exp(-rf * T) * cdf(-d1)
    }
}

/// Calculate the volatility of an option that is implied by the price.
#[allow(non_snake_case)]
pub fn ivol(
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    rf: f64,
    p: f64,
    max_iterations: usize,
    epsilon: f64,
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, rf, v),
        max_iterations,
        epsilon,
    )
}

/// Return a struct to calculate greeks numerically using finite difference methods.
pub fn fdm_greeks(is_call: bool) -> NumericGreeks {
    #[allow(non_snake_case)]
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, rf: f64, v: f64| {
        price(is_call, S, K, T, r, rf, v)
    })
}
