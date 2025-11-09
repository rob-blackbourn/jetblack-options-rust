
use core::f64;

use libm::{erf, exp, sqrt};

/// Cumulative distribution function.  P(X <= x)
pub fn cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    
    if sigma < 0.0 {
        return f64::NAN; // sigma must be non-negative
    }

    if sigma == 0.0 {
        return f64::NAN; // Err("cdf() not defined when sigma is zero");
    }

    0.5 * (1.0 + erf((x - mu) / (sigma * f64::consts::SQRT_2)))
}

/// Probability density function.  P(x <= X < x+dx) / dx
pub fn pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    
    if sigma < 0.0 {
        return f64::NAN; // sigma must be non-negative
    }

    let variance = sigma * sigma;
    if variance == 0.0 {
        return f64::NAN; // pdf() not defined when sigma is zero
    }

    let diff = x - mu;

    exp(diff * diff / (-2.0 * variance)) / sqrt(f64::consts::TAU * variance)
}
