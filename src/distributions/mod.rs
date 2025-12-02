pub mod binomial_coefficient;
pub mod cbnd;
pub mod cdf;
pub mod chi_inv;
pub mod cndev;
pub mod inv_cdf;
pub mod pdf;

pub use inv_cdf::inv_cdf;

pub fn cdf(x: f64) -> f64 {
    cdf::cumulative_distribution_function(x, 0.0, 1.0)
}

pub fn pdf(x: f64) -> f64 {
    pdf::probability_density_function(x, 0.0, 1.0)
}
