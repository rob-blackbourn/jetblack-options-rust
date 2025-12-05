use crate::distributions::cdf;

/// Black Scholes variance analytic solutions
pub struct BlackScholes {}

impl BlackScholes {
    /// The generalized Black and Scholes formula on variance form.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + v / 2.0) * t) / (v * t).sqrt();
        let d2 = d1 - (v * t).sqrt();

        if is_call {
            S * ((b - r) * t).exp() * cdf(d1) - K * (-r * t).exp() * cdf(d2)
        } else {
            K * (-r * t).exp() * cdf(-d2) - S * ((b - r) * t).exp() * cdf(-d1)
        }
    }
}
