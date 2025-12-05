//! # Black-Scholes-Merton options pricing formulae.
//!
//! This is the "generalised" version using "cost of carry" (variable b).
//!
//! The cost of carry rate (b) is:
//!
//! * b == r: for non dividend paying stocks
//! * b == r - q: For dividend paying stocks where the dividend yield is q
//! * b == 0: for futures options
//! * b = r - rj: for currency options.
//!
//! The following arguments are common.
//!
//! * is_call (bool): True for a call, false for a put.
//! * S (f64): The current asset price.
//! * K (f64): The option strike price
//! * t (f64): The time to expiry of the option in years.
//! * r (f64): The risk free rate.
//! * b (f64): The cost of carry of the asset.
//! * v (f64): The volatility of the asset.
//! * max_iterations (int, Optional): The maximum number of iterations before a price is returned. Defaults to 20.
//! * epsilon (f64, Optional): The largest acceptable error. Defaults to 1e-8.

use std::f64::consts::PI;

use crate::distributions::{cdf, inv_cdf, pdf};
use crate::fdm::FdmWithCarry;
use crate::implied_volatility::solve_ivol;

pub struct GeneralizedBlackScholes {}

impl GeneralizedBlackScholes {
    /// The fair value of a European option, using Black-Scholes-Merton.
    #[allow(non_snake_case)]
    pub fn price(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + t * (b + (v * v) / 2.0)) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        if is_call {
            S * ((b - r) * t).exp() * cdf(d1) - K * (-r * t).exp() * cdf(d2)
        } else {
            K * (-r * t).exp() * cdf(-d2) - S * ((b - r) * t).exp() * cdf(-d1)
        }
    }

    /// Calculate the volatility of an option that is implied by the price.
    #[allow(non_snake_case)]
    pub fn ivol(
        is_call: bool,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        b: f64,
        p: f64,
        max_iterations: usize,
        epsilon: f64,
    ) -> f64 {
        solve_ivol(
            p,
            |v| GeneralizedBlackScholes::price(is_call, S, K, t, r, b, v),
            max_iterations,
            epsilon,
        )
    }

    /// Return a struct to calculate greeks numerically using finite difference methods.
    pub fn fdm_greeks<'a>(is_call: bool) -> FdmWithCarry<'a> {
        #[allow(non_snake_case)]
        FdmWithCarry::new(move |S: f64, K: f64, t: f64, r: f64, b: f64, v: f64| {
            GeneralizedBlackScholes::price(is_call, S, K, t, r, b, v)
        })
    }

    /// The sensitivity of the option to a change in the asset price.
    #[allow(non_snake_case)]
    pub fn delta(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + t * (b + (v * v) / 2.0)) / (v * t.sqrt());

        if is_call {
            ((b - r) * t).exp() * cdf(d1)
        } else {
            -((b - r) * t).exp() * cdf(-d1)
        }
    }

    /// The second derivative to the change in the asset price.
    #[allow(non_snake_case)]
    pub fn gamma(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + t * (b + (v * v) / 2.0)) / (v * t.sqrt());

        ((b - r) * t).exp() * pdf(d1) / (S * v * t.sqrt())
    }

    /// The theta or time decay of the value of the option.
    #[allow(non_snake_case)]
    pub fn theta(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        if is_call {
            let p1 = -S * ((b - r) * t).exp() * pdf(d1) * v / (2.0 * t.sqrt());
            let p2 = (b - r) * S * ((b - r) * t).exp() * cdf(d1);
            let p3 = r * K * (-r * t).exp() * cdf(d2);

            p1 - p2 - p3
        } else {
            let p1 = -S * ((b - r) * t).exp() * pdf(d1) * v / (2.0 * t.sqrt());
            let p2 = (b - r) * S * ((b - r) * t).exp() * cdf(-d1);
            let p3 = r * K * (-r * t).exp() * cdf(-d2);

            p1 + p2 + p3
        }
    }

    /// The sensitivity of the options price or a change in the asset volatility.
    ///
    /// This value is typically reported by dividing by 100 (for a 1% change in
    /// volatility)
    #[allow(non_snake_case)]
    pub fn vega(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        S * ((b - r) * t).exp() * pdf(d1) * t.sqrt()
    }

    /// The sensitivity of the option price to the risk free rate.
    ///
    /// Useful for all options except futures options which should use
    /// futures_rho.
    #[allow(non_snake_case)]
    pub fn rho(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        if is_call {
            t * K * (-r * t).exp() * cdf(d2)
        } else {
            -t * K * (-r * t).exp() * cdf(-d2)
        }
    }

    /// Sensitivity to the cost of carry.
    #[allow(non_snake_case)]
    pub fn carry(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        if is_call {
            t * S * ((b - r) * t).exp() * cdf(d1)
        } else {
            -t * S * ((b - r) * t).exp() * cdf(-d1)
        }
    }

    /// The percentage change in the option price for a percentage change in the
    /// asset price.
    ///
    /// This is thought of as a measure of leverage, sometimes called gearing.
    ///
    /// Also known as lambda or omega.
    #[allow(non_snake_case)]
    pub fn elasticity(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        return GeneralizedBlackScholes::delta(is_call, S, K, t, r, b, v) * S
            / GeneralizedBlackScholes::price(is_call, S, K, t, r, b, v);
    }

    #[allow(non_snake_case)]
    pub fn gammap(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        GeneralizedBlackScholes::gamma(S, K, t, r, b, v) * S / 100.0
    }

    #[allow(non_snake_case)]
    pub fn vegap(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        GeneralizedBlackScholes::vega(S, K, t, r, b, v) * v * 10.0
    }

    #[allow(non_snake_case)]
    pub fn forward_delta(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());

        if is_call {
            (-r * t).exp() * cdf(d1)
        } else {
            (-r * t).exp() * (cdf(d1) - 1.0)
        }
    }

    /// The second order derivative of the option price to a change in the asset
    /// price and a change in the volatility.
    ///
    /// Also known as DdeltaDvol.
    #[allow(non_snake_case)]
    pub fn vanna(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        -((b - r) * t).exp() * d2 / v * pdf(d1)
    }

    /// Also known as DVannaDvol
    #[allow(non_snake_case)]
    pub fn ddelta_dvol_dvol(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        GeneralizedBlackScholes::vanna(S, K, t, r, b, v) / v * (d1 * d2 - d1 / d2 - 1.0)
    }

    /// Measures the instantaneous rate of change of delta over the passage of
    /// time.
    ///
    /// Also known as DdeltaDtime.
    #[allow(non_snake_case)]
    pub fn charm(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        if is_call {
            -((b - r) * t).exp()
                * (pdf(d1) * (b / (v * t.sqrt()) - d2 / (2.0 * t)) + (b - r) * cdf(d1))
        } else {
            return -((b - r) * t).exp()
                * (pdf(d1) * (b / (v * t.sqrt()) - d2 / (2.0 * t)) - (b - r) * cdf(-d1));
        }
    }

    #[allow(non_snake_case)]
    pub fn saddle_gamma(K: f64, r: f64, b: f64, v: f64) -> f64 {
        return (f64::exp(1.0) / PI).sqrt() * ((2.0 * b - r) / (v * v) + 1.0).sqrt() / K;
    }

    /// Also known as Speed
    #[allow(non_snake_case)]
    pub fn dgamma_dspot(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        -GeneralizedBlackScholes::gamma(S, K, t, r, b, v) * (1.0 + d1 / (v * t.sqrt())) / S
    }

    /// Also known as zomma.
    #[allow(non_snake_case)]
    pub fn dgamma_dvol(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        GeneralizedBlackScholes::gamma(S, K, t, r, b, v) * ((d1 * d2 - 1.0) / v)
    }

    #[allow(non_snake_case)]
    pub fn dgamma_dtime(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        GeneralizedBlackScholes::gamma(S, K, t, r, b, v)
            * (r - b + b * d1 / (v * t.sqrt()) + (1.0 - d1 * d2) / (2.0 * t))
    }

    /// Also known as SpeedP.
    #[allow(non_snake_case)]
    pub fn dgammap_dspot(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        return -GeneralizedBlackScholes::gamma(S, K, t, r, b, v) * (d1) / (100.0 * v * t.sqrt());
    }

    #[allow(non_snake_case)]
    pub fn dgammap_dvol(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        S / 100.0 * GeneralizedBlackScholes::gamma(S, K, t, r, b, v) * ((d1 * d2 - 1.0) / v)
    }

    #[allow(non_snake_case)]
    pub fn dgammap_dtime(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        GeneralizedBlackScholes::gammap(S, K, t, r, b, v)
            * (r - b + b * d1 / (v * t.sqrt()) + (1.0 - d1 * d2) / (2.0 * t))
    }

    #[allow(non_snake_case)]
    pub fn dvega_dtime(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        GeneralizedBlackScholes::vega(S, K, t, r, b, v)
            * (r - b + b * d1 / (v * t.sqrt()) - (1.0 + d1 * d2) / (2.0 * t))
    }

    /// Also known as DvegaDvol
    #[allow(non_snake_case)]
    pub fn vomma(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        GeneralizedBlackScholes::vega(S, K, t, r, b, v) * d1 * d2 / v
    }

    #[allow(non_snake_case)]
    pub fn dvomma_dvol(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        GeneralizedBlackScholes::vomma(S, K, t, r, b, v) * 1.0 / v
            * (d1 * d2 - d1 / d2 - d2 / d1 - 1.0)
    }

    /// Also known as VommaP.
    #[allow(non_snake_case)]
    pub fn dvegap_dvol(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        return GeneralizedBlackScholes::vegap(S, K, t, r, b, v) * d1 * d2 / v;
    }

    #[allow(non_snake_case)]
    pub fn vega_leverage(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        GeneralizedBlackScholes::vega(S, K, t, r, b, v) * v
            / GeneralizedBlackScholes::price(is_call, S, K, t, r, b, v)
    }

    #[allow(non_snake_case)]
    pub fn variance_vega(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        S * ((b - r) * t).exp() * pdf(d1) * t.sqrt() / (2.0 * v)
    }

    #[allow(non_snake_case)]
    pub fn variance_delta(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        S * ((b - r) * t).exp() * pdf(d1) * (-d2) / (2.0 * (v * v))
    }

    #[allow(non_snake_case)]
    pub fn variance_vomma(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        S * ((b - r) * t).exp() * t.sqrt() / (4.0 * (v * v * v)) * pdf(d1) * (d1 * d2 - 1.0)
    }

    #[allow(non_snake_case)]
    pub fn variance_ultima(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        S * ((b - r) * t).exp() * t.sqrt() / (8.0 * (v * v * v * v * v))
            * pdf(d1)
            * ((d1 * d2 - 1.0) * (d1 * d2 - 3.0) - ((d1 * d1) + (d2 * d2)))
    }

    #[allow(non_snake_case)]
    pub fn theta_driftless(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        -S * ((b - r) * t).exp() * pdf(d1) * v / (2.0 * t.sqrt())
    }

    #[allow(non_snake_case)]
    pub fn futures_rho(is_call: bool, S: f64, K: f64, t: f64, r: f64, v: f64) -> f64 {
        -t * GeneralizedBlackScholes::price(is_call, S, K, t, r, 0.0, v)
    }

    /// Also known as rho2.
    #[allow(non_snake_case)]
    pub fn phi(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        if is_call {
            -t * S * ((b - r) * t).exp() * cdf(d1)
        } else {
            t * S * ((b - r) * t).exp() * cdf(-d1)
        }
    }

    #[allow(non_snake_case)]
    pub fn dzeta_dvol(is_call: bool, S: f64, K: f64, t: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        if is_call {
            -pdf(d2) * d1 / v
        } else {
            pdf(d2) * d1 / v
        }
    }

    #[allow(non_snake_case)]
    pub fn dzeta_dtime(is_call: bool, S: f64, K: f64, t: f64, b: f64, v: f64) -> f64 {
        let d1 = ((S / K).ln() + (b + (v * v) / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        if is_call {
            return pdf(d2) * (b / (v * t.sqrt()) - d1 / (2.0 * t));
        } else {
            return -pdf(d2) * (b / (v * t.sqrt()) - d1 / (2.0 * t));
        }
    }

    /// Risk neutral break even probability.
    #[allow(non_snake_case)]
    pub fn break_even_probability(
        is_call: bool,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        b: f64,
        v: f64,
    ) -> f64 {
        if is_call {
            let K = K + GeneralizedBlackScholes::price(true, S, K, t, r, b, v) * (r * t).exp();
            let d2 = ((S / K).ln() + (b - (v * v) / 2.0) * t) / (v * t.sqrt());
            return cdf(d2);
        } else {
            let K = K - GeneralizedBlackScholes::price(false, S, K, t, r, b, v) * (r * t).exp();
            let d2 = ((S / K).ln() + (b - (v * v) / 2.0) * t) / (v * t.sqrt());
            return cdf(-d2);
        }
    }

    #[allow(non_snake_case)]
    pub fn strike_delta(is_call: bool, S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d2 = ((S / K).ln() + (b - (v * v) / 2.0) * t) / (v * t.sqrt());
        if is_call {
            -(-r * t).exp() * cdf(d2)
        } else {
            (-r * t).exp() * cdf(-d2)
        }
    }

    #[allow(non_snake_case)]
    pub fn risk_neutral_density(S: f64, K: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
        let d2 = ((S / K).ln() + (b - (v * v) / 2.0) * t) / (v * t.sqrt());
        (-r * t).exp() * pdf(d2) / (K * v * t.sqrt())
    }

    #[allow(non_snake_case)]
    pub fn gamma_from_delta(S: f64, t: f64, r: f64, b: f64, v: f64, delta_: f64) -> f64 {
        ((b - r) * t).exp() * pdf(inv_cdf(((r - b) * t).exp() * delta_.abs())) / (S * v * t.sqrt())
    }

    #[allow(non_snake_case)]
    pub fn gammap_from_delta(S: f64, t: f64, r: f64, b: f64, v: f64, delta_: f64) -> f64 {
        S / 100.0 * GeneralizedBlackScholes::gamma_from_delta(S, t, r, b, v, delta_)
    }

    #[allow(non_snake_case)]
    pub fn vega_from_delta(S: f64, t: f64, r: f64, b: f64, delta_: f64) -> f64 {
        S * ((b - r) * t).exp() * t.sqrt() * pdf(inv_cdf(((r - b) * t).exp() * delta_.abs()))
    }

    #[allow(non_snake_case)]
    pub fn vegap_from_delta(S: f64, t: f64, r: f64, b: f64, v: f64, delta_: f64) -> f64 {
        v / 10.0 * GeneralizedBlackScholes::vega_from_delta(S, t, r, b, delta_)
    }

    #[allow(non_snake_case)]
    pub fn strike_from_delta(
        is_call: bool,
        S: f64,
        t: f64,
        r: f64,
        b: f64,
        v: f64,
        delta_: f64,
    ) -> f64 {
        if is_call {
            S * (-inv_cdf(delta_ * ((r - b) * t).exp()) * v * t.sqrt() + (b + v * v / 2.0) * t)
                .exp()
        } else {
            S * (inv_cdf(-delta_ * ((r - b) * t).exp()) * v * t.sqrt() + (b + v * v / 2.0) * t)
                .exp()
        }
    }

    #[allow(non_snake_case)]
    pub fn in_the_money_prob_from_delta(
        is_call: bool,
        t: f64,
        r: f64,
        b: f64,
        v: f64,
        delta_: f64,
    ) -> f64 {
        if is_call {
            cdf(inv_cdf(delta_ / ((b - r) * t).exp()) - v * t.sqrt())
        } else {
            cdf(inv_cdf(-delta_ / ((b - r) * t).exp()) + v * t.sqrt())
        }
    }

    #[allow(non_snake_case)]
    pub fn strike_from_in_the_money_prob(
        is_call: bool,
        S: f64,
        v: f64,
        t: f64,
        b: f64,
        in_the_money_prob: f64,
    ) -> f64 {
        if is_call {
            return S
                * (-inv_cdf(in_the_money_prob) * v * t.sqrt() + (b - (v * v) / 2.0) * t).exp();
        } else {
            return S * (inv_cdf(in_the_money_prob) * v * t.sqrt() + (b - (v * v) / 2.0) * t).exp();
        }
    }

    #[allow(non_snake_case)]
    pub fn rnd_from_in_the_money_prob(
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        in_the_money_prob: f64,
    ) -> f64 {
        (-r * t).exp() * pdf(inv_cdf(in_the_money_prob)) / (K * v * t.sqrt())
    }

    #[allow(non_snake_case)]
    pub fn delta_from_in_the_money_prob(
        is_call: bool,
        t: f64,
        r: f64,
        b: f64,
        v: f64,
        in_the_money_prob: f64,
    ) -> f64 {
        if is_call {
            return cdf(inv_cdf(in_the_money_prob * ((b - r) * t).exp()) - v * t.sqrt());
        } else {
            return -cdf(inv_cdf(in_the_money_prob * ((b - r) * t).exp()) + v * t.sqrt());
        }
    }

    /// What asset price that gives maximum DdeltaDvol
    ///
    /// is_lower == True gives lower asset level that gives max DdeltaDvol
    /// is_lower == False gives upper asset level that gives max DdeltaDvol
    #[allow(non_snake_case)]
    pub fn max_ddelta_dvol_asset(is_lower: bool, K: f64, t: f64, b: f64, v: f64) -> f64 {
        if is_lower {
            return K * (-b * t - v * t.sqrt() * (4.0 + t * (v * v)).sqrt() / 2.0).exp();
        } else {
            return K * (-b * t + v * t.sqrt() * (4.0 + t * (v * v)).sqrt() / 2.0).exp();
        }
    }

    /// What strike price that gives maximum DdeltaDvol
    ///
    /// is_lower == True gives lower strike level that gives max DdeltaDvol
    /// is_lower == False gives upper strike level that gives max DdeltaDvol
    #[allow(non_snake_case)]
    pub fn max_ddelta_dvol_strike(is_lower: bool, S: f64, t: f64, b: f64, v: f64) -> f64 {
        if is_lower {
            S * (b * t - v * t.sqrt() * (4.0 + t * (v * v)).sqrt() / 2.0).exp()
        } else {
            S * (b * t + v * t.sqrt() * (4.0 + t * (v * v)).sqrt() / 2.0).exp()
        }
    }

    /// What strike price that gives maximum gamma and vega
    #[allow(non_snake_case)]
    pub fn max_gamma_vega_at_X(S: f64, b: f64, t: f64, v: f64) -> f64 {
        S * ((b + (v * v) / 2.0) * t).exp()
    }

    /// What asset price that gives maximum gamma
    #[allow(non_snake_case)]
    pub fn max_gamma_at_S(x: f64, b: f64, t: f64, v: f64) -> f64 {
        x * ((-b - 3.0 * (v * v) / 2.0) * t).exp()
    }

    /// What asset price that gives maximum vega
    #[allow(non_snake_case)]
    pub fn max_vega_at_S(K: f64, b: f64, t: f64, v: f64) -> f64 {
        K * ((-b + (v * v) / 2.0) * t).exp()
    }

    #[allow(non_snake_case)]
    pub fn in_the_money_probability(is_call: bool, S: f64, K: f64, t: f64, b: f64, v: f64) -> f64 {
        let d2 = ((S / K).ln() + (b - (v * v) / 2.0) * t) / (v * t.sqrt());

        if is_call { cdf(d2) } else { cdf(-d2) }
    }

    #[allow(non_snake_case)]
    pub fn delta_mirror_strike(S: f64, t: f64, b: f64, v: f64) -> f64 {
        return S * ((b + (v * v) / 2.0) * t).exp();
    }

    #[allow(non_snake_case)]
    pub fn probability_mirror_strike(S: f64, t: f64, b: f64, v: f64) -> f64 {
        S * ((b - (v * v) / 2.0) * t).exp()
    }

    #[allow(non_snake_case)]
    pub fn delta_mirror_call_put_strike(S: f64, K: f64, t: f64, b: f64, v: f64) -> f64 {
        (S * S) / K * ((2.0 * b + (v * v)) * t).exp()
    }

    #[allow(non_snake_case)]
    pub fn profit_loss_std(
        is_cash: bool,
        is_call: bool,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        b: f64,
        v: f64,
        n_hedges: i32,
    ) -> f64 {
        if is_cash {
            (PI / 4.0).sqrt() * GeneralizedBlackScholes::vega(S, K, t, r, b, v) * v
                / (n_hedges as f64).sqrt()
        } else {
            (PI / 4.0).sqrt() * GeneralizedBlackScholes::vega(S, K, t, r, b, v) * v
                / (n_hedges as f64).sqrt()
                / GeneralizedBlackScholes::price(is_call, S, K, t, r, b, v)
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f64;
    use std::collections::HashMap;

    use crate::fdm::DifferenceMethod;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = (actual - expected).abs();
        diff < threshold
    }

    #[test]
    fn it_should_calc_price() {
        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                11.069546131685598,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.50565027500145199,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                3.8695002999527546,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.913498834791838,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.78816855802530128,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                9.344461337871536,
            ),
        ] {
            let b = r - q;
            let actual = GeneralizedBlackScholes::price(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(actual, expected, f64::EPSILON),
                "price({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                actual,
                expected,
                (expected - actual).abs(),
                f64::EPSILON
            );
        }
    }

    #[test]
    fn it_should_calc_ivol() {
        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, p, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                11.069546131685598,
                0.125,
                1e-12,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.505650275001452,
                0.125,
                1e-12,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                3.8695002999527546,
                0.125,
                1e-12,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                2.913498834791845,
                0.125,
                1e-12,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.7881685580252977,
                0.125,
                1e-12,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                9.344461337871536,
                0.125,
                1e-12,
            ),
        ] {
            let b = r - q;
            let actual =
                GeneralizedBlackScholes::ivol(is_call, S, K, t, r, b, p, 100, f64::EPSILON / 2.0);
            assert!(
                is_close_to(actual, expected, threshold),
                "ivol({}, {}, {}, {}, {}, {}, {}) -> {} ( expected: {}, diff: {:e}, expected: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                p,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_delta() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.8567400985874144,
                1e-10,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.10404934056490878,
                1e-11,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.54045184861735829,
                1e-10,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.42033759053496483,
                1e-10,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.17153007262292191,
                1e-10,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.78925936652940132,
                1e-10,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::delta(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, f64::EPSILON),
                "delta({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                f64::EPSILON
            );

            let numeric = ng[&is_call].delta(S, K, t, r, b, v, 1e-4, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].delta({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_gamma() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018374151835767315,
                1e-8,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.018374151835767315,
                1e-8,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
                1e-8,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
                1e-8,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910805,
                1e-8,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910805,
                1e-8,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::gamma(S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, f64::EPSILON),
                "gamma({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                threshold
            );

            let numeric = ng[&is_call].gamma(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].gamma({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_theta() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.5148051444486299,
                1e-8,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.4574579639819345,
                1e-8,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -4.0402024708588584,
                1e-8,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.2142237390703023,
                1e-8,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -2.4811528460769701,
                1e-10,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.29605531021229758,
                1e-8,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::theta(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, f64::EPSILON),
                "theta({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                threshold
            );

            let numeric = ng[&is_call].theta(
                S,
                K,
                t,
                r,
                b,
                v,
                1.0 / 365.0 / 24.0 / 60.0,
                DifferenceMethod::Central,
            );
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].theta({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_vega() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.895452325799033,
                1e-7,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.895452325799033,
                1e-7,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.769990428955332,
                1e-8,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                26.769990428955332,
                1e-9,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.73527645306925,
                1e-7,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.73527645306925,
                1e-7,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::vega(S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, f64::EPSILON),
                "vega({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                threshold
            );

            let numeric = ng[&is_call].vega(S, K, t, r, b, v, 1e-5, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].vega({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_rho() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                41.585932356464994,
                1e-8,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -5.9755388685707089,
                1e-8,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                25.08784228089154,
                1e-8,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -22.47362894414416,
                1e-8,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                8.1824193521334454,
                1e-8,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -44.13519899540583,
                1e-8,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::rho(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, f64::EPSILON),
                "rho({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                threshold
            );

            let numeric = ng[&is_call].rho(S, K, t, r, b, v, 1e-6, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].rho({}, {}, {}, {}, {}, {}) -> {} (diff={:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic - numeric
            );
        }
    }

    #[test]
    fn it_should_calc_elasticity() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                8.51357496716671,
                1e-4,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -22.635066226567563,
                1e-4,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                13.966967482182572,
                1e-4,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -14.427244161400045,
                1e-4,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                21.763120448838848,
                1e-4,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -8.446279972615066,
                1e-4,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::elasticity(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, 1e-12),
                "elasticity({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                f64::EPSILON
            );

            let numeric =
                ng[&is_call].elasticity(S, K, t, r, b, v, 1e-4, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].elasticity({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (expected - analytic).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_dgamma_dvol() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.06138389948689551,
                1e-4,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.06138389948689551,
                1e-4,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.338939132019472,
                1e-4,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.338939132019472,
                1e-4,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.01597963672325094,
                1e-4,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.01597963672325094,
                1e-4,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::dgamma_dvol(S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, 1e-12),
                "dgamma_dvol({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                threshold
            );

            let numeric =
                ng[&is_call].dgamma_dvol(S, K, t, r, b, v, 0.01, 0.001, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].dgamma_dvol({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_gammap() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.020211567019344047,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.020211567019344047,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.042831984686328525,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910798,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.028376442324910798,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::gammap(S, K, t, r, b, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].gammap(S, K, t, r, b, v, 0.01, DifferenceMethod::Central);
            assert!(is_close_to(numeric, analytic, 1e-5));
        }
    }

    #[test]
    fn it_should_calc_vanna() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.639625858611978,
                1e-6,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -1.639625858611978,
                1e-6,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.2088059253458516,
                1e-6,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.2088059253458516,
                1e-6,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.0253158998215026,
                1e-7,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.0253158998215026,
                1e-7,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::vanna(S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, 1e-12),
                "vanna({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                1e-12
            );

            let numeric =
                ng[&is_call].vanna(S, K, t, r, b, v, 0.001, 0.0001, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].vanna({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_charm() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.23306930617480232,
                1e-8,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                0.15620615104261645,
                1e-8,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.01632708081503694,
                1e-7,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.0931902359472228,
                1e-7,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.2961949663176757,
                1e-9,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                -0.37305812144986156,
                1e-7,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::charm(is_call, S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, 1e-12),
                "charm({}, {}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                threshold
            );

            let numeric = ng[&is_call].charm(
                S,
                K,
                t,
                r,
                b,
                v,
                0.001,
                1.0 / 365.0 / 60.0,
                DifferenceMethod::Central,
            );
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].charm({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }

    #[test]
    fn it_should_calc_vegap() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.36931540724879,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                17.36931540724879,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                33.462488036194166,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                33.462488036194166,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                22.169095566336562,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                22.169095566336562,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::vegap(S, K, t, r, b, v);
            assert!(is_close_to(analytic, expected, 1e-12));

            let numeric = ng[&is_call].vegap(S, K, t, r, b, v, 0.001, DifferenceMethod::Central);
            assert!(is_close_to(numeric, analytic, 1e-3));
        }
    }

    #[test]
    fn it_should_calc_vomma() {
        let ng = HashMap::from([
            (true, GeneralizedBlackScholes::fdm_greeks(true)),
            (false, GeneralizedBlackScholes::fdm_greeks(false)),
        ]);

        #[allow(non_snake_case)]
        for (is_call, S, K, r, q, t, v, expected, threshold) in [
            (
                true,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                157.585192593357,
                1e-5,
            ),
            (
                false,
                110.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                157.585192593357,
                1e-4,
            ),
            (
                true,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.32296591947259933,
                1e-7,
            ),
            (
                false,
                100.0,
                100.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                2.3229659194725993,
                1e-6,
            ),
            (
                true,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                131.8949386725222,
                1e-4,
            ),
            (
                false,
                100.0,
                110.0,
                0.1,
                0.08,
                6.0 / 12.0,
                0.125,
                131.8949386725222,
                1e-4,
            ),
        ] {
            let b = r - q;
            let analytic = GeneralizedBlackScholes::vomma(S, K, t, r, b, v);
            assert!(
                is_close_to(analytic, expected, 1e-12),
                "vomma({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                S,
                K,
                t,
                r,
                b,
                v,
                analytic,
                expected,
                (expected - analytic).abs(),
                1e-12
            );

            let numeric = ng[&is_call].vomma(S, K, t, r, b, v, 0.0001, DifferenceMethod::Central);
            assert!(
                is_close_to(numeric, analytic, threshold),
                "[{}].vomma({}, {}, {}, {}, {}, {}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                is_call,
                S,
                K,
                t,
                r,
                b,
                v,
                numeric,
                analytic,
                (analytic - numeric).abs(),
                threshold
            );
        }
    }
}
