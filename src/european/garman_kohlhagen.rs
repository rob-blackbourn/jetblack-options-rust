//! # Garman and Kohlhagen (1983) Currency options.
//!
//! The value of a call option.
//!
//! $$
//! c = S_0e^{-r_f t}\mathcal{N}(d_1) - Ke^{-r_d t}\mathcal{N}(d_2)
//! $$
//!
//! The value of a put option.
//!
//! $$
//! p = Ke^{-r_d t}\mathcal{N}(-d_2) - S_0e^{-r_f t}\mathcal{N}(-d_1)
//! $$
//!
//! where:
//!
//! $$
//! d_1 = \frac{\ln(S_0/K) + (r_d - r_f + \sigma^2/2)t}{\sigma\sqrt{t}}
//! $$
//!
//! and
//!
//! $$
//! d_2 = d_1 - \sigma\sqrt{t}
//! $$
//!
//! * $S_0$ is the current spot rate
//! * $K$ is the strike price
//! * $\mathcal{N}(x)$ is the cumulative normal distribution function
//! * $r_d$ is domestic risk free [[simple interest]] rate
//! * $r_f$ is foreign risk free simple interest rate
//! * $t$ is the time to maturity (calculated according to the appropriate day count convention)
//! * $\sigma$ is the volatility of the FX rate.
//!
//! Command arguments are:
//!
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The asset price.
//! * K (f64): The strike price.
//! * t (f64): The time to expiry in years.
//! * r (f64): The risk free rate of the base currency.
//! * rf (f64): The risk free rate of the quote currency.
//! * v (f64): The asset volatility.
//! * max_iterations (int, Optional): The maximum number of iterations before a price is returned. Defaults to 20.
//! * epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.

use crate::distributions::cdf;
use crate::fdm::FdmWithDividendYield;
use crate::implied_volatility::solve_ivol;

pub struct GarmanKohlhagen {}

impl GarmanKohlhagen {
    /// The fair value of a currency option.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, rf: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (r - rf + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        if is_call {
            S * (-rf * t).exp() * cdf(d1) - K * (-r * t).exp() * cdf(d2)
        } else {
            K * (-r * t).exp() * cdf(-d2) - S * (-rf * t).exp() * cdf(-d1)
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
        rf: f64,
        p: f64,
        max_iterations: usize,
        epsilon: f64,
    ) -> f64 {
        solve_ivol(
            p,
            |v| GarmanKohlhagen::price(is_call, S, K, t, r, rf, v),
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks<'a>(is_call: bool) -> FdmWithDividendYield<'a> {
        #[allow(non_snake_case)]
        FdmWithDividendYield::new(move |S: f64, K: f64, t: f64, r: f64, rf: f64, v: f64| {
            GarmanKohlhagen::price(is_call, S, K, t, r, rf, v)
        })
    }
}
