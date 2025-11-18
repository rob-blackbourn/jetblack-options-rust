use core::f64;

use libm::erf;

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
