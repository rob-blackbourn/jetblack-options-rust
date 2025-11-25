use core::f64::consts::TAU;

use libm::{exp, sqrt};

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

    exp(diff * diff / (-2.0 * variance)) / sqrt(TAU * variance)
}
