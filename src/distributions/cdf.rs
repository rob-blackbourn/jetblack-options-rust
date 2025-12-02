use core::f64;

use libm::erf;

/// Cumulative distribution function.  P(X <= x)
pub fn cumulative_distribution_function(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma < 0.0 {
        return f64::NAN; // sigma must be non-negative
    }

    if sigma == 0.0 {
        return f64::NAN; // Err("cumulative_distribution_function() not defined when sigma is zero");
    }

    0.5 * (1.0 + erf((x - mu) / (sigma * f64::consts::SQRT_2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_should_calc_cdf() {
        for (x, mu, sigma, expected) in [
            (0.0, 0.0, 1.0, 0.5),
            (0.1, 0.0, 1.0, 0.539827837277029),
            (0.2, 0.0, 1.0, 0.579259709439103),
            (0.3, 0.0, 1.0, 0.6179114221889526),
            (0.4, 0.0, 1.0, 0.6554217416103242),
            (0.5, 0.0, 1.0, 0.6914624612740131),
            (0.6, 0.0, 1.0, 0.7257468822499265),
            (0.7, 0.0, 1.0, 0.758036347776927),
            (0.8, 0.0, 1.0, 0.7881446014166034),
            (0.9, 0.0, 1.0, 0.8159398746532405),
            (1.0, 0.0, 1.0, 0.8413447460685429),
        ] {
            let actual = cumulative_distribution_function(x, mu, sigma);
            assert!(
                (expected - actual).abs() <= f64::EPSILON,
                "cumulative_distribution_function({}, {}, {}) -> {} (expected: {}, diff: {:e})",
                x,
                mu,
                sigma,
                actual,
                expected,
                (expected - actual).abs()
            )
        }
    }
}
