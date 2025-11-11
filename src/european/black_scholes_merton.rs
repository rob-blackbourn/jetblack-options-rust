/// Black-Scholes-Merton options pricing formulae using dividend yield.
/// 
/// * Stock price $ S $,
/// * Strike price $ K $,
/// * Risk-free rate $ r $,
/// * Annual dividend yield $ q $,
/// * Time to maturity $ \tau = T - t $
/// * Volatility $ \sigma $.
/// 
/// where:
/// 
/// $$
/// d_1 = \frac{\ln(S/K) + \left(r - q + \frac{1}{2}\sigma^2\right)\tau}{\sigma\sqrt{\tau}}
/// $$
/// 
/// $$
/// d_2 = \frac{\ln(S/K) + \left(r - q - \frac{1}{2}\sigma^2\right)\tau}{\sigma\sqrt{\tau}} = d_1 - \sigma\sqrt{\tau}
/// $$
/// 
/// $$
/// \varphi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2}
/// $$
/// 
/// $$
/// \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{1}{2} y^2} \,dy = 1 - \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-\frac{1}{2} y^2} \,dy
/// $$

use libm::{exp, log, pi, sqrt};

use crate::{implied_volatility::solve_ivol, numeric_greeks::with_dividend_yield::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}
