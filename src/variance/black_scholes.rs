//! Black Scholes variance analytic solutions

use libm::{exp, log, sqrt};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}

/// The generalized Black and Scholes formula on variance form.
#[allow(non_snake_case)]
pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = (log(S / K) + (b + v / 2.0) * t) / sqrt(v * t);
    let d2 = d1 - sqrt(v * t);

    if is_call {
        S * exp((b - r) * t) * cdf(d1) - K * exp(-r * t) * cdf(d2)
    } else {
        K * exp(-r * t) * cdf(-d2) - S * exp((b - r) * t) * cdf(-d1)
    }
}
