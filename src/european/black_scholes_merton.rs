use core::f64;

/// Black-Scholes-Merton options pricing formulae using dividend yield.
///
/// * Stock price $ S $,
/// * Strike price $ K $,
/// * Risk-free rate $ r $,
/// * Annual dividend yield $ q $,
/// * Time to maturity $ \tau = T - t $
/// * Volatility $ \sigma $.
///
/// where:
///
/// $$
/// d_1 = \frac{\ln(S/K) + \left(r - q + \frac{1}{2}\sigma^2\right)\tau}{\sigma\sqrt{\tau}}
/// $$
///
/// $$
/// d_2 = \frac{\ln(S/K) + \left(r - q - \frac{1}{2}\sigma^2\right)\tau}{\sigma\sqrt{\tau}} = d_1 - \sigma\sqrt{\tau}
/// $$
///
/// $$
/// \varphi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2}
/// $$
///
/// $$
/// \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{1}{2} y^2} \,dy = 1 - \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-\frac{1}{2} y^2} \,dy
/// $$
use libm::{exp, fabs, log, sqrt};

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
/// Call price: $Se^{-q \tau}\Phi(d_1) - e^{-r \tau} K\Phi(d_2)$
///
/// Put price: $e^{-r \tau} K\Phi(-d_2) -  Se^{-q \tau}\Phi(-d_1)$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The price of the options.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    let F = S * exp((r - q) * T);
    if is_call {
        exp(-r * T) * (F * cdf(d1) - K * cdf(d2))
    } else {
        exp(-r * T) * (K * cdf(-d2) - F * cdf(-d1))
    }
}

/// Calculate the volatility of an option that is implied by the price.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
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
    q: f64,
    p: f64,
    max_iterations: Option<i32>, // = 20,
    epsilon: Option<f64>,        // =1e-8
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, S, K, T, r, q, v),
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
    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, q: f64, v: f64| {
        price(is_call, S, K, T, r, q, v)
    })
}

/// The sensitivity of the open to a change in the asset price.
///
/// Call $\Delta$  $e^{-q \tau} \Phi(d_1)$
///
/// Put $\Delta$ $-e^{-q \tau} \Phi(-d_1)$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: the delta.
#[allow(non_snake_case)]
pub fn delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));

    if is_call {
        exp(-q * T) * cdf(d1)
    } else {
        -exp(-q * T) * cdf(-d1)
    }
}

/// The second derivative to the change in the asset price.
///
/// $$
/// \Gamma $ $ e^{-q \tau} \frac{\varphi(d_1)}{S\sigma\sqrt{\tau}} = K e^{-r \tau} \frac{\varphi(d_2)}{S^2\sigma\sqrt{\tau}}
/// $$
///
/// Args:
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The gamma.
#[allow(non_snake_case)]
pub fn gamma(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));

    exp(-q * T) * pdf(d1) / (S * v * sqrt(T))
}

/// The theta or time decay of the value of the option.
///
/// $$
/// Call \Theta $ $ - e^{-q \tau} \frac{S \varphi(d_1) \sigma}{2 \sqrt{\tau}} - rKe^{-r \tau}\Phi(d_2) + qSe^{-q \tau}\Phi(d_1)
/// $$
///
/// $$
/// Put \Theta $ $ - e^{-q \tau}\frac{S \varphi(d_1) \sigma}{2 \sqrt{\tau}} + rKe^{-r \tau}\Phi(-d_2) - qSe^{-q \tau}\Phi(-d_1)
/// $$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The theta.
#[allow(non_snake_case)]
pub fn theta(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        let p1 = -S * exp(-q * T) * pdf(d1) * v / (2.0 * sqrt(T));
        let p2 = -q * S * exp(-q * T) * cdf(d1);
        let p3 = r * K * exp(-r * T) * cdf(d2);

        p1 - p2 - p3
    } else {
        let p1 = -S * exp(-q * T) * pdf(d1) * v / (2.0 * sqrt(T));
        let p2 = -q * S * exp(-q * T) * cdf(-d1);
        let p3 = r * K * exp(-r * T) * cdf(-d2);

        p1 + p2 + p3
    }
}

/// The sensitivity of the options price or a change in the asset volatility.
///
/// $$
/// \mathcal{V} $ is $ S e^{-q \tau} \varphi(d_1) \sqrt{\tau} = K e^{-r \tau} \varphi(d_2) \sqrt{\tau}
/// $$
///
/// Args:
///     S (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The volatility of the asset.
///
/// Returns:
///     f64: The vega
#[allow(non_snake_case)]
pub fn vega(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    S * exp(-q * T) * pdf(d1) * sqrt(T)
}

/// The sensitivity of the option price to the risk free rate.
///
/// Call $ \rho $ is $ K \tau e^{-r \tau}\Phi(d_2) $
///
/// Put $ \rho $ is $ -K \tau e^{-r \tau}\Phi(-d_2) $
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
///     q (f64): The dividend yield.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The rho.
#[allow(non_snake_case)]
pub fn rho(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        K * T * exp(-r * T) * cdf(d2)
    } else {
        -K * T * exp(-r * T) * cdf(-d2)
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
///     q (f64): The dividend yield.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The carry.
#[allow(non_snake_case)]
pub fn carry(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    if is_call {
        T * S * exp(-q * T) * cdf(d1)
    } else {
        -T * S * exp(-q * T) * cdf(-d1)
    }
}

/// The option elasticity.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The elasticity.
#[allow(non_snake_case)]
pub fn elasticity(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    delta(is_call, S, K, T, r, q, v) * S / price(is_call, S, K, T, r, q, v)
}

#[allow(non_snake_case)]
pub fn gammap(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    S * gamma(S, K, T, r, q, v) / 100.0
}

#[allow(non_snake_case)]
pub fn vegap(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    v / 10.0 * vega(S, K, T, r, q, v)
}

#[allow(non_snake_case)]
pub fn forward_delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));

    if is_call {
        exp(-r * T) * cdf(d1)
    } else {
        exp(-r * T) * (cdf(d1) - 1.0)
    }
}

/// The sensitivity to the spot price and volatility.
///
/// $$
/// -e^{-q \tau} \varphi(d_1) \frac{d_2}{\sigma} \, = \frac{\mathcal{V}}{S}\left[1 - \frac{d_1}{\sigma\sqrt{\tau}} \right]
/// $$
///
/// Also known as DdeltaDvol.
///
/// Args:
///     S (f64): The asset price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     q (f64): The dividend yield.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The vanna.
#[allow(non_snake_case)]
pub fn vanna(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as DdeltaDvol.
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    -exp(-q * T) * d2 / v * pdf(d1)
}

#[allow(non_snake_case)]
pub fn ddelta_dvol_dvol(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as DVannaDvol
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vanna(S, K, T, r, q, v) * 1.0 / v * (d1 * d2 - d1 / d2 - 1.0)
}

#[allow(non_snake_case)]
pub fn charm(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as DdeltaDtime

    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        q * exp(-q * T) * cdf(d1)
            - exp(-q * T) * pdf(d1) * (2.0 * (r - q) * T - d2 * v * sqrt(T))
                / (2.0 * T * v * sqrt(T))
    } else {
        -q * exp(-q * T) * cdf(-d1)
            - exp(-q * T) * pdf(d1) * (2.0 * (r - q) * T - d2 * v * sqrt(T))
                / (2.0 * T * v * sqrt(T))
    }
}

#[allow(non_snake_case)]
pub fn saddle_gamma(K: f64, q: f64, v: f64) -> f64 {
    sqrt(exp(1.0) / f64::consts::PI) * sqrt((2.0 * -q) / (v * v) + 1.0) / K
}

#[allow(non_snake_case)]
pub fn dgamma_dspot(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as Speed
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    -gamma(S, K, T, r, q, v) * (1.0 + d1 / (v * sqrt(T))) / S
}

#[allow(non_snake_case)]
pub fn dgamma_dvol(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as zomma.
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    gamma(S, K, T, r, q, v) * ((d1 * d2 - 1.0) / v)
}

#[allow(non_snake_case)]
pub fn dgamma_dtime(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    gamma(S, K, T, r, q, v) * (q + (r - q) * d1 / (v * sqrt(T)) + (1.0 - d1 * d2) / (2.0 * T))
}

#[allow(non_snake_case)]
pub fn dgammap_dspot(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as SpeedP.
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    -gamma(S, K, T, r, q, v) * (d1) / (100.0 * v * sqrt(T))
}

#[allow(non_snake_case)]
pub fn dgammap_dvol(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S / 100.0 * gamma(S, K, T, r, q, v) * ((d1 * d2 - 1.0) / v)
}

#[allow(non_snake_case)]
pub fn dgammap_dtime(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    gammap(S, K, T, r, q, v) * (q + (r - q) * d1 / (v * sqrt(T)) + (1.0 - d1 * d2) / (2.0 * T))
}

#[allow(non_snake_case)]
pub fn dvega_dtime(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vega(S, K, T, r, q, v) * (q + (r - q) * d1 / (v * sqrt(T)) - (1.0 + d1 * d2) / (2.0 * T))
}

#[allow(non_snake_case)]
pub fn vomma(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as DvegaDvol
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vega(S, K, T, r, q, v) * d1 * d2 / v
}

#[allow(non_snake_case)]
pub fn dvomma_dvol(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vomma(S, K, T, r, q, v) * 1.0 / v * (d1 * d2 - d1 / d2 - d2 / d1 - 1.0)
}

#[allow(non_snake_case)]
pub fn dvegap_dvol(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as VommaP.
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    vegap(S, K, T, r, q, v) * d1 * d2 / v
}

#[allow(non_snake_case)]
pub fn vega_leverage(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    vega(S, K, T, r, q, v) * v / price(is_call, S, K, T, r, q, v)
}

#[allow(non_snake_case)]
pub fn variance_vega(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    S * exp(-q * T) * pdf(d1) * sqrt(T) / (2.0 * v)
}

#[allow(non_snake_case)]
pub fn variance_delta(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S * exp(-q * T) * pdf(d1) * (-d2) / (2.0 * (v * v))
}

#[allow(non_snake_case)]
pub fn variance_vomma(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S * exp(-q * T) * sqrt(T) / (4.0 * (v * v * v)) * pdf(d1) * (d1 * d2 - 1.0)
}

#[allow(non_snake_case)]
pub fn variance_ultima(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    S * exp(-q * T) * sqrt(T) / (8.0 * (v * v * v * v * v))
        * pdf(d1)
        * ((d1 * d2 - 1.0) * (d1 * d2 - 3.0) - ((d1 * d1) + (d2 * d2)))
}

#[allow(non_snake_case)]
pub fn theta_driftless(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    -S * exp(-q * T) * pdf(d1) * v / (2.0 * sqrt(T))
}

#[allow(non_snake_case)]
pub fn futures_rho(is_call: bool, S: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    -T * price(is_call, S, K, T, r, 0.0, v)
}

#[allow(non_snake_case)]
pub fn phi(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // Also known as rho2.
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    if is_call {
        -T * S * exp(-q * T) * cdf(d1)
    } else {
        T * S * exp(-q * T) * cdf(-d1)
    }
}

#[allow(non_snake_case)]
pub fn dzeta_dvol(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        -pdf(d2) * d1 / v
    } else {
        pdf(d2) * d1 / v
    }
}

#[allow(non_snake_case)]
pub fn dzeta_dtime(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + T * (r - q + (v * v) / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        pdf(d2) * ((r - q) / (v * sqrt(T)) - d1 / (2.0 * T))
    } else {
        -pdf(d2) * ((r - q) / (v * sqrt(T)) - d1 / (2.0 * T))
    }
}

#[allow(non_snake_case)]
pub fn break_even_probability(
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    q: f64,
    v: f64,
) -> f64 {
    // Risk neutral break even probability.
    if is_call {
        let K = K + price(true, S, K, T, r, q, v) * exp(r * T);
        let d2 = (log(S / K) + (r - q - (v * v) / 2.0) * T) / (v * sqrt(T));
        cdf(d2)
    } else {
        let K = K - price(false, S, K, T, r, q, v) * exp(r * T);
        let d2 = (log(S / K) + (r - q - (v * v) / 2.0) * T) / (v * sqrt(T));
        cdf(-d2)
    }
}

#[allow(non_snake_case)]
pub fn strike_delta(is_call: bool, S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d2 = (log(S / K) + (r - q - (v * v) / 2.0) * T) / (v * sqrt(T));
    if is_call {
        -exp(-r * T) * cdf(d2)
    } else {
        exp(-r * T) * cdf(-d2)
    }
}

#[allow(non_snake_case)]
pub fn risk_neutral_density(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    let d2 = (log(S / K) + (r - q - (v * v) / 2.0) * T) / (v * sqrt(T));
    exp(-r * T) * pdf(d2) / (K * v * sqrt(T))
}

#[allow(non_snake_case)]
pub fn gamma_from_delta(S: f64, T: f64, q: f64, v: f64, delta_: f64) -> f64 {
    exp(-q * T) * pdf(inv_cdf(exp(q * T) * fabs(delta_))) / (S * v * sqrt(T))
}

#[allow(non_snake_case)]
pub fn gammap_from_delta(S: f64, T: f64, q: f64, v: f64, delta_: f64) -> f64 {
    S / 100.0 * gamma_from_delta(S, T, q, v, delta_)
}

#[allow(non_snake_case)]
pub fn vega_from_delta(S: f64, T: f64, q: f64, delta_: f64) -> f64 {
    S * exp(-q * T) * sqrt(T) * pdf(inv_cdf(exp(q * T) * fabs(delta_)))
}

#[allow(non_snake_case)]
pub fn vegap_from_delta(S: f64, T: f64, q: f64, v: f64, delta_: f64) -> f64 {
    v / 10.0 * vega_from_delta(S, T, q, delta_)
}

#[allow(non_snake_case)]
pub fn strike_from_delta(
    is_call: bool,
    S: f64,
    T: f64,
    r: f64,
    q: f64,
    v: f64,
    delta_: f64,
) -> f64 {
    if is_call {
        S * exp(-inv_cdf(delta_ * exp(q * T)) * v * sqrt(T) + (r - q + v * v / 2.0) * T)
    } else {
        S * exp(inv_cdf(-delta_ * exp(q * T)) * v * sqrt(T) + (r - q + v * v / 2.0) * T)
    }
}

#[allow(non_snake_case)]
pub fn in_the_money_prob_from_delta(
    is_call: bool,
    T: f64,
    r: f64,
    q: f64,
    v: f64,
    delta_: f64,
) -> f64 {
    if is_call {
        cdf(inv_cdf(delta_ / exp(-q * T)) - v * sqrt(T))
    } else {
        cdf(inv_cdf(-delta_ / exp(-q * T)) + v * sqrt(T))
    }
}

#[allow(non_snake_case)]
pub fn strike_from_in_the_money_prob(
    is_call: bool,
    S: f64,
    v: f64,
    T: f64,
    r: f64,
    q: f64,
    in_the_money_prob: f64,
) -> f64 {
    if is_call {
        S * exp(-inv_cdf(in_the_money_prob) * v * sqrt(T) + (r - q - v * v / 2.0) * T)
    } else {
        S * exp(inv_cdf(in_the_money_prob) * v * sqrt(T) + (r - q - v * v / 2.0) * T)
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
    q: f64,
    v: f64,
    in_the_money_prob: f64,
) -> f64 {
    if is_call {
        cdf(inv_cdf(in_the_money_prob * exp(-q * T)) - v * sqrt(T))
    } else {
        -cdf(inv_cdf(in_the_money_prob * exp(-q * T)) + v * sqrt(T))
    }
}

#[allow(non_snake_case)]
pub fn max_ddelta_dvol_asset(is_lower: bool, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // What asset price that gives maximum DdeltaDvol
    //
    // is_lower == True gives lower asset level that gives max DdeltaDvol
    // is_lower == False gives upper asset level that gives max DdeltaDvol

    if is_lower {
        K * exp((q - r) * T - v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0)
    } else {
        K * exp((q - r) * T + v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0)
    }
}

#[allow(non_snake_case)]
pub fn max_ddelta_dvol_strike(is_lower: bool, S: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    // What strike price that gives maximum DdeltaDvol
    //
    // is_lower == True gives lower strike level that gives max DdeltaDvol
    // is_lower == False gives upper strike level that gives max DdeltaDvol

    if is_lower {
        S * exp((r - q) * T - v * sqrt(T) * sqrt(4.0 + T * v * 2.0) / 2.0)
    } else {
        S * exp((r - q) * T + v * sqrt(T) * sqrt(4.0 + T * (v * v)) / 2.0)
    }
}

#[allow(non_snake_case)]
pub fn max_gamma_vega_at_X(S: f64, r: f64, q: f64, T: f64, v: f64) -> f64 {
    // What strike price that gives maximum gamma and vega
    S * exp((r - q + v * v / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn max_gamma_at_S(x: f64, r: f64, q: f64, T: f64, v: f64) -> f64 {
    // What asset price that gives maximum gamma
    x * exp((q - r - 3.0 * v * v / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn max_vega_at_S(K: f64, r: f64, q: f64, T: f64, v: f64) -> f64 {
    // What asset price that gives maximum vega
    K * exp((q - r + v * v / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn in_the_money_probability(
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    q: f64,
    v: f64,
) -> f64 {
    let d2 = (log(S / K) + (r - q - (v * v) / 2.0) * T) / (v * sqrt(T));

    if is_call { cdf(d2) } else { cdf(-d2) }
}

#[allow(non_snake_case)]
pub fn delta_mirror_strike(S: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    S * exp((r - q + (v * v) / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn probability_mirror_strike(S: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    S * exp((r - q - (v * v) / 2.0) * T)
}

#[allow(non_snake_case)]
pub fn delta_mirror_call_put_strike(S: f64, K: f64, T: f64, r: f64, q: f64, v: f64) -> f64 {
    (S * S) / K * exp((2.0 * (r - q) + (v * v)) * T)
}

#[allow(non_snake_case)]
pub fn profit_loss_std(
    is_absolute: bool,
    is_call: bool,
    S: f64,
    K: f64,
    T: f64,
    r: f64,
    q: f64,
    v: f64,
    n_hedges: i32,
) -> f64 {
    if is_absolute {
        // as a value
        sqrt(f64::consts::PI / 4.0) * vega(S, K, T, r, q, v) * v / sqrt(n_hedges.into())
    } else {
        // as a percent
        sqrt(f64::consts::PI / 4.0) * vega(S, K, T, r, q, v) * v
            / sqrt(n_hedges.into())
            / price(is_call, S, K, T, r, q, v)
    }
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
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.069546131685598,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.505650275001452,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8695002999527546,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.913498834791845,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.7881685580252977,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                9.344461337871536,
            ),
        ] {
            let actual = price(is_call, S, K, T, r, q, v);
            assert!(is_close_to(actual, expected, 1e-12));
        }
    }

    #[test]
    fn it_should_calc_ivol() {
        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, T, p, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                11.069546131685598,
                0.125,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.505650275001452,
                0.125,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                3.8695002999527546,
                0.125,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                2.913498834791845,
                0.125,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.7881685580252977,
                0.125,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                9.344461337871536,
                0.125,
            ),
        ] {
            let actual = ivol(is_call, S, K, T, r, q, p, None, None);
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
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.8567400985874144,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.10404934056490878,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.5404518486173583,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.42033759053496483,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17153007262292186,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.7892593665294013,
            ),
        ] {
            let analytic = delta(is_call, S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].delta(S, K, T, r, q, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-5));
        }
    }

    #[test]
    fn it_should_calc_gamma() {
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
                0.018374151835767315,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018374151835767315,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910798,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910798,
            ),
        ] {
            let analytic = gamma(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].gamma(S, K, T, r, q, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-5));
        }
    }

    #[test]
    fn it_should_calc_theta() {
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
                -2.514805144448628,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.457457963981934,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -4.040202470858858,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.2142237390703023,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.48115284607697,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.2960553102122976,
            ),
        ] {
            let analytic = theta(is_call, S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].theta(S, K, T, r, q, v, None, None);
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
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.895452325799033,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.895452325799033,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.76999042895533,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.76999042895533,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.73527645306925,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.73527645306925,
            ),
        ] {
            let analytic = vega(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vega(S, K, T, r, q, v, None, None);
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
        for (is_call, S, K, r, q, T, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                41.58593235646499,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -5.975538868570712,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                25.08784228089154,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -22.47362894414416,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                8.182419352133445,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -44.13519899540583,
            ),
        ] {
            let analytic = rho(is_call, S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].rho(S, K, T, r, q, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_elasticity() {
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
                8.51357496716671,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -22.635066226567563,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.966967482182572,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -14.427244161400045,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                21.763120448838848,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -8.446279972615066,
            ),
        ] {
            let analytic = elasticity(is_call, S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].elasticity(S, K, T, r, q, v, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_dgamma_dvol() {
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
                0.06138389948689551,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.06138389948689551,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.338939132019472,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.338939132019472,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.01597963672325094,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.01597963672325094,
            ),
        ] {
            let analytic = dgamma_dvol(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].dgamma_dvol(S, K, T, r, q, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-4));
        }
    }

    #[test]
    fn it_should_calc_gammap() {
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
                0.020211567019344047,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.020211567019344047,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910798,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910798,
            ),
        ] {
            let analytic = gammap(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].gammap(S, K, T, r, q, v, None);
            assert!(is_close_to(numeric, analytic, 1e-5));
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
                -1.639625858611978,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.639625858611978,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.2088059253458516,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.2088059253458516,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.0253158998215026,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.0253158998215026,
            ),
        ] {
            let analytic = vanna(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vanna(S, K, T, r, q, v, None, None);
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
                0.23306930617480232,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.15620615104261645,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.01632708081503694,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.0931902359472228,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.2961949663176757,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.37305812144986156,
            ),
        ] {
            let analytic = charm(is_call, S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].charm(S, K, T, r, q, v, None, None);
            assert!(is_close_to(numeric, analytic, 1e-5));
        }
    }

    #[test]
    fn it_should_calc_vegap() {
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
                0.17369315407248792,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17369315407248792,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.33462488036194166,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.33462488036194166,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.22169095566336564,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.22169095566336564,
            ),
        ] {
            let analytic = vegap(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vegap(S, K, T, r, q, v, None);
            assert!(is_close_to(numeric, analytic, 1e-3));
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
                157.585192593357,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                157.585192593357,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.3229659194725993,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.3229659194725993,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                131.8949386725222,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                131.8949386725222,
            ),
        ] {
            let analytic = vomma(S, K, T, r, q, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vomma(S, K, T, r, q, v, None);
            assert!(is_close_to(numeric, analytic, 1e-2));
        }
    }
}
