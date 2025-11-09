/// Black (1976) Options on futures/forwards
///
///
/// * The discounted futures price $ F $,
/// * Strike price $ K $,
/// * Risk-free rate $ r $,
/// * Annual dividend yield $ q $,
/// * Time to maturity $ \tau = T - t $
/// * Volatility $ \sigma $.
///
/// Most of the formula use one or both of the following terms.
///
/// $$
/// d_1 = \frac{\ln(F/K) + (\sigma^2/2)T}{\sigma\sqrt{T}}
/// $$
///
/// $$
/// d_2 = \frac{\ln(F/K) - (\sigma^2/2)T}{\sigma\sqrt{T}} = d_1 - \sigma\sqrt{T}
/// $$
///
/// $$
/// \varphi(x) &= \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2}
/// $$
///
/// $$
/// \Phi(x) &= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{1}{2} y^2} \,dy = 1 - \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-\frac{1}{2} y^2} \,dy
/// $$
use libm::{exp, log, sqrt};

use crate::{implied_volatility::solve_ivol, numeric_greeks::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}
fn pdf(x: f64) -> f64 {
    crate::distributions::pdf(x, 0.0, 1.0)
}

/// Fair value of a futures/forward using Black 76.
///
/// For a call:
///
/// $$
/// C = e^{-r \tau}[F\Phi(d_1) - K\Phi(d_2)]
/// $$
///
/// For a put:
///
/// $$
/// P = e^{-r \tau} [K\Phi(-d_2) -  F\Phi(-d_1)]
/// $$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     F (f64): The price of the future.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The option price.
#[allow(non_snake_case)]
pub fn price(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        exp(-r * T) * (F * cdf(d1) - K * cdf(d2))
    } else {
        exp(-r * T) * (K * cdf(-d2) - F * cdf(-d1))
    }
}

/// Calculate the volatility of a Black 76 option that is implied by the price.
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     F (f64): The current asset price.
///     K (f64): The option strike price
///     T (f64): The time to maturity of the option in years.
///     r (f64): The risk free rate.
///     p (f64): The option price.
///     max_iterations (int, Optional): The maximum number of iterations before
///         a price is returned. Defaults to 20.
///     epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.
///
/// Returns:
///     f64: The implied volatility.
#[allow(non_snake_case)]
pub fn ivol(
    is_call: bool,
    F: f64,
    K: f64,
    T: f64,
    r: f64,
    p: f64,
    max_iterations: Option<i32>,
    epsilon: Option<f64>,
) -> f64 {
    solve_ivol(
        p,
        |v| price(is_call, F, K, T, r, v),
        max_iterations,
        epsilon,
    )
}

/// Make a class to generate greeks numerically using finite difference methods.
///
/// Args:
///     is_call (bool): If true the options is a call;  otherwise it is a put.
///
/// Returns:
///     NumericGreeks: A class which can generate Greeks using finite difference
///         methods.
pub fn make_numeric_greeks<T>(is_call: bool) -> NumericGreeks
where
    T: Fn(f64, f64, f64, f64, f64) -> f64,
{
    // Normalize the price function to match that required by the finite
    // difference methods.

    NumericGreeks::new(move |S: f64, K: f64, T: f64, r: f64, b: f64| price(is_call, S, K, T, r, b))
}

/// The sensitivity of the option to a change in the asset price
/// using Black 76.
///
/// For the call.
///
/// $$
/// \frac{\partial C}{\partial S} = e^{-r \tau} \Phi(d_1)
/// $$
///
/// For the put.
///
/// $$
/// \frac{\partial P}{\partial S} = -e^{-r \tau} \Phi(-d_1)
/// $$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     F (f64): The current futures price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The volatility.
///
/// Returns:
///     f64: The delta.
#[allow(non_snake_case)]
pub fn delta(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));
    if is_call {
        return exp(-r * T) * cdf(d1);
    } else {
        return -exp(-r * T) * cdf(-d1);
    }
}

/// The second derivative to the change in asset price using Black 76.
///
/// The gamma for both calls and puts.
///
/// $$
/// \frac{\partial^2 V}{\partial S^2} = e^{-r \tau} \frac{\varphi(d_1)}{F\sigma\sqrt{\tau}} = K e^{-r \tau} \frac{\varphi(d_2)}{F^2\sigma\sqrt{\tau}}
/// $$
///
/// Args:
///     F (f64): The current futures price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The volatility.
///
/// Returns:
///     f64: The gamma.
#[allow(non_snake_case)]
pub fn gamma(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));

    exp(-r * T) * pdf(d1) / (F * v * sqrt(T))
}

/// The change in the value of the option with respect to time to expiry
/// using Black 76.
///
/// For the call.
///
/// $$
/// \frac{\partial C}{\partial T} - \frac{F e^{-r \tau} \varphi(d_1) \sigma}{2 \sqrt{\tau}} - rKe^{-r \tau}\Phi(d_2) + rFe^{-r \tau}\Phi(d_1)
/// $$
///
/// For the put.
///
/// $$
/// \frac{\partial P}{\partial T} - \frac{F e^{-r \tau} \varphi(d_1) \sigma}{2 \sqrt{\tau}} + rKe^{-r \tau}\Phi(-d_2) - rFe^{-r \tau}\Phi(-d_1)
/// $$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     F (f64): The current futures price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The volatility.
///
/// Returns:
///     f64: The theta.
#[allow(non_snake_case)]
pub fn theta(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    if is_call {
        -F * exp(-r * T) * pdf(d1) * v / (2.0 * sqrt(T)) + r * F * exp(-r * T) * cdf(d1)
            - r * K * exp(-r * T) * cdf(d2)
    } else {
        -F * exp(-r * T) * pdf(d1) * v / (2.0 * sqrt(T)) - r * F * exp(-r * T) * cdf(-d1)
            + r * K * exp(-r * T) * cdf(-d2)
    }
}

/// The sensitivity of the options price or a change in the asset volatility
/// using Black 76.
///
/// For both calls and puts.
///
/// $$
/// \frac{\partial V}{\partial \sigma} = F e^{-r \tau} \varphi(d_1) \sqrt{\tau} = K e^{-r \tau} \varphi(d_2) \sqrt{\tau}
/// $$
///
/// Args:
///     F (f64): The current futures price.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The volatility.
///
/// Returns:
///     f64: The vega.
#[allow(non_snake_case)]
pub fn vega(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));

    F * exp(-r * T) * pdf(d1) * sqrt(T)
}

/// The sensitivity of the option price to a change in the risk free rate
/// using Black 76.
///
/// For a call:
///
/// $$
/// \frac{\partial C}{\partial r} = -\tau e^{-r \tau}[F\Phi(d_1) - K\Phi(d_2)]
/// $$
///
/// For a put:
///
/// $$
/// \frac{\partial P}{\partial r} = -\tau e^{-r \tau} [K\Phi(-d_2) -  F\Phi(-d_1)]
/// $$
///
/// Args:
///     is_call (bool): True for a call, false for a put.
///     F (f64): The price of the future.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The rho.
#[allow(non_snake_case)]
pub fn rho(is_call: bool, F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + (v * v / 2.0) * T) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);
    if is_call {
        -T * exp(-r * T) * (F * cdf(d1) - K * cdf(d2))
    } else {
        -T * exp(-r * T) * (K * cdf(-d2) - F * cdf(-d1))
    }
}

/// The sensitivity of the option value to the underlying
/// asset price and the volatility.
///
/// For both calls and puts.
///
/// $$
/// \frac{\partial^2 V}{\partial F \partial \sigma} = -e^{-r \tau} \varphi(d_1) \frac{d_2}{\sigma} \, = \frac{\mathcal{V}}{F}\left[1 - \frac{d_1}{\sigma\sqrt{\tau}} \right]
/// $$
///
/// Args:
///     F (f64): The price of the future.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The vanna.
#[allow(non_snake_case)]
pub fn vanna(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    -exp(-r * T) * pdf(d1) * d2 / v
}

/// The second order sensitivity to volatility.
///
/// For both puts and calls.
///
/// $$
/// \frac{\partial^2 V}{\partial \sigma^2} = F e^{-r \tau} \varphi(d_1) \sqrt{\tau} \frac{d_1 d_2}{\sigma} = \mathcal{V}  \frac{d_1 d_2}{\sigma}
/// $$
///
/// Args:
///     F (f64): The price of the future.
///     K (f64): The strike price.
///     T (f64): The time to expiry in years.
///     r (f64): The risk free rate.
///     v (f64): The asset volatility.
///
/// Returns:
///     f64: The vomma
#[allow(non_snake_case)]
pub fn vomma(F: f64, K: f64, T: f64, r: f64, v: f64) -> f64 {
    let d1 = (log(F / K) + T * (v * v / 2.0)) / (v * sqrt(T));
    let d2 = d1 - v * sqrt(T);

    F * exp(-r * T) * pdf(d1) * sqrt(T) * d1 * d2 / v
}

#[cfg(test)]
mod tests {
    use libm::fabs;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        fabs(actual - expected) < threshold
    }

    #[test]
    fn it_should_calc_price() {
        for (is_call, F, K, T, r, v, expected) in [
            (
                true,
                110.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                10.143390791460092,
            ),
            (
                false,
                110.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                0.6310965464529535,
            ),
            (
                true,
                100.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                3.3531192847248605,
            ),
            (
                false,
                100.0,
                100.0,
                6.0 / 12.0,
                0.1,
                0.125,
                3.3531192847248534,
            ),
            (
                true,
                100.0,
                110.0,
                6.0 / 12.0,
                0.1,
                0.125,
                0.6310965464529654,
            ),
            (
                false,
                100.0,
                110.0,
                6.0 / 12.0,
                0.1,
                0.125,
                10.143390791460092,
            ),
        ] {
            let actual = price(is_call, F, K, T, r, v);
            assert!(is_close_to(actual, expected, 1e-12));
        }
    }

    #[test]
    fn it_should_calc_ivol() {
        for (is_call, F, K, r, T, p, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                10.143390791460092,
                0.125,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                6.0 / 12.0,
                0.6310965464529535,
                0.125,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                3.3531192847248605,
                0.125,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                6.0 / 12.0,
                3.3531192847248534,
                0.125,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                0.6310965464529654,
                0.125,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                6.0 / 12.0,
                10.143390791460092,
                0.125,
            ),
        ] {
            let actual = ivol(is_call, F, K, T, r, p, None, None);
            assert!(is_close_to(actual, expected, 1e-9));
        }
    }
}
