use core::f64;

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

    exp(diff * diff / (-2.0 * variance)) / sqrt(f64::consts::TAU * variance)
}

fn r8poly_value(n: usize, a: &[f64], x: f64) -> f64 {
    let mut value = 0.0;

    for i in (0..n).rev() {
        value = value * x + a[i];
    }

    return value;
}
