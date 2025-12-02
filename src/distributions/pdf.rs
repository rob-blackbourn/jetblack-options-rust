use core::f64::consts::TAU;

/// Probability density function.  P(x <= X < x+dx) / dx
pub fn probability_density_function(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma < 0.0 {
        return f64::NAN; // sigma must be non-negative
    }

    let variance = sigma * sigma;
    if variance == 0.0 {
        return f64::NAN; // pdf() not defined when sigma is zero
    }

    let diff = x - mu;

    (diff * diff / (-2.0 * variance)).exp() / (TAU * variance).sqrt()
}
