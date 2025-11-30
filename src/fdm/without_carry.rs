//! Calculating numeric greeks for options using finite difference methods for pricers without carry or dividend yield.
//!
//! The following arguments are common.
//!
//! * S (f64): The asset price.
//! * K (f64): the strike price.
//! * t (f64): The time to expiry in years.
//! * r (f64): the risk free rate.
//! * v (f64): The volatility.
//! * dS (f64): The asset price bump. A common choice is 0.01.
//! * method (DifferenceMethod): The difference method. A common choice is Central.

use crate::fdm::DifferenceMethod;

/// A struct for calculating option sensitivities using finite difference methods for pricers without carry or dividend yield.
pub struct FdmWithoutCarry {
    /// fn price(S: 64, K: f64, t, r: f64, v: f64) -> f64
    pub price: Box<dyn Fn(f64, f64, f64, f64, f64) -> f64>,
}

impl FdmWithoutCarry {
    pub fn new(price: impl Fn(f64, f64, f64, f64, f64) -> f64 + 'static) -> Self {
        FdmWithoutCarry {
            price: Box::new(price),
        }
    }

    #[allow(non_snake_case)]
    pub fn delta(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        method: DifferenceMethod,
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, t, r, v) - (self.price)(S - dS, K, t, r, v)) / dS
            }
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, t, r, v) - (self.price)(S - dS, K, t, r, v)) / (2.0 * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + dS, K, t, r, v) - (self.price)(S, K, t, r, v)) / dS
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn gamma(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        method: DifferenceMethod,
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, t, r, v) - 2.0 * (self.price)(S - dS, K, t, r, v)
                    + (self.price)(S - 2.0 * dS, K, t, r, v))
                    / (dS * dS)
            }
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, t, r, v) - 2.0 * (self.price)(S, K, t, r, v)
                    + (self.price)(S - dS, K, t, r, v))
                    / (dS * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + 2.0 * dS, K, t, r, v) - 2.0 * (self.price)(S + dS, K, t, r, v)
                    + (self.price)(S, K, t, r, v))
                    / (dS * dS)
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn theta(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dT: f64,
        method: DifferenceMethod,
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, t - dT, r, v) - (self.price)(S, K, t, r, v)) / dT
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, t - dT, r, v) - (self.price)(S, K, t + dT, r, v)) / (2.0 * dT)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, t, r, v) - (self.price)(S, K, t + dT, r, v)) / dT
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn vega(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dv: f64,
        method: DifferenceMethod,
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, t, r, v) - (self.price)(S, K, t, r, v - dv)) / dv
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, t, r, v + dv) - (self.price)(S, K, t, r, v - dv)) / (2.0 * dv)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, t, r, v + dv) - (self.price)(S, K, t, r, v)) / dv
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn rho(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dr: f64,
        method: DifferenceMethod,
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, t, r, v) - (self.price)(S, K, t, r - dr, v)) / dr
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, t, r + dr, v) - (self.price)(S, K, t, r - dr, v)) / (2.0 * dr)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, t, r + dr, v) - (self.price)(S, K, t, r, v)) / dr
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn elasticity(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        method: DifferenceMethod,
    ) -> f64 {
        self.delta(S, K, t, r, v, dS, method) * S / (self.price)(S, K, t, r, v)
    }

    #[allow(non_snake_case)]
    pub fn speed(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S + 2.0 * dS, K, t, r, v) - 3.0 * (self.price)(S + dS, K, t, r, v)
            + 3.0 * (self.price)(S, K, t, r, v)
            - (self.price)(S - dS, K, t, r, v))
            / (dS * dS * dS)
    }

    #[allow(non_snake_case)]
    pub fn deltap(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S * (1.0 + dS), K, t, r, v) - (self.price)(S * (1.0 - dS), K, t, r, v))
            / (2.0 * S)
    }

    #[allow(non_snake_case)]
    pub fn gammap(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        method: DifferenceMethod,
    ) -> f64 {
        S / 100.0 * self.gamma(S, K, t, r, v, dS, method)
    }

    #[allow(non_snake_case)]
    pub fn vegap(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dv: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K, t, r, v + dv) - (self.price)(S, K, t, r, v - dv)) * v / 0.1 / 2.0
    }

    /// Also known as DdeltaDvol
    #[allow(non_snake_case)]
    pub fn vanna(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        dv: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S + dS, K, t, r, v + dv)
            - (self.price)(S + dS, K, t, r, v - dv)
            - (self.price)(S - dS, K, t, r, v + dv)
            + (self.price)(S - dS, K, t, r, v - dv))
            / (4.0 * dS)
            / dv
    }

    /// Also known as DdeltaDtime
    #[allow(non_snake_case)]
    pub fn charm(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        dT: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S + dS, K, t + dT, r, v)
            - (self.price)(S + dS, K, t - dT, r, v)
            - (self.price)(S - dS, K, t + dT, r, v)
            + (self.price)(S - dS, K, t - dT, r, v))
            / (4.0 * dS)
            / -dT
    }

    #[allow(non_snake_case)]
    pub fn dgamma_dvol(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dS: f64,
        dv: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S + dS, K, t, r, v + dv) - 2.0 * (self.price)(S, K, t, r, v + dv)
            + (self.price)(S - dS, K, t, r, v + dv)
            - (self.price)(S + dS, K, t, r, v - dv)
            + 2.0 * (self.price)(S, K, t, r, v - dv)
            - (self.price)(S - dS, K, t, r, v - dv))
            / (2.0 * dv * (dS * dS))
            / 100.0
    }

    /// Also known as DvegaDvol
    #[allow(non_snake_case)]
    pub fn vomma(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dv: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K, t, r, v + dv) - 2.0 * (self.price)(S, K, t, r, v)
            + (self.price)(S, K, t, r, v - dv))
            / (dv * dv)
    }

    #[allow(non_snake_case)]
    pub fn time_gamma(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dT: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K, t + dT, r, v) - 2.0 * (self.price)(S, K, t, r, v)
            + (self.price)(S, K, t - dT, r, v))
            / (dT * dT)
    }

    #[allow(non_snake_case)]
    pub fn strike_delta(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dK: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K + dK, t, r, v) - (self.price)(S, K - dK, t, r, v)) / (2.0 * dK)
    }

    #[allow(non_snake_case)]
    pub fn strike_gamma(
        &self,
        S: f64,
        K: f64,
        t: f64,
        r: f64,
        v: f64,
        dK: f64,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K + dK, t, r, v) - 2.0 * (self.price)(S, K, t, r, v)
            + (self.price)(S, K - dK, t, r, v))
            / (dK * dK)
    }
}
