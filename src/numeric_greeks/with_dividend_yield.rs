//! # Class for calculating numeric greeks for options using finite difference
//!
//! methods for the dividend yield style.
use crate::numeric_greeks::DifferenceMethod;

pub struct NumericGreeks {
    /// fn price(S: 64, K: f64, T, r: f64, q: f64, v: f64) -> f64
    pub price: Box<dyn Fn(f64, f64, f64, f64, f64, f64) -> f64>,
}

impl NumericGreeks {
    pub fn new(price: impl Fn(f64, f64, f64, f64, f64, f64) -> f64 + 'static) -> Self {
        NumericGreeks {
            price: Box::new(price),
        }
    }

    #[allow(non_snake_case)]
    pub fn delta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                  // = 0.01,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, q, v) - (self.price)(S - dS, K, T, r, q, v)) / dS
            }
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, T, r, q, v) - (self.price)(S - dS, K, T, r, q, v))
                    / (2.0 * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + dS, K, T, r, q, v) - (self.price)(S, K, T, r, q, v)) / dS
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                  // = 0.01,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, q, v) - 2.0 * (self.price)(S - dS, K, T, r, q, v)
                    + (self.price)(S - 2.0 * dS, K, T, r, q, v))
                    / (dS * dS)
            }
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, T, r, q, v) - 2.0 * (self.price)(S, K, T, r, q, v)
                    + (self.price)(S - dS, K, T, r, q, v))
                    / (dS * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + 2.0 * dS, K, T, r, q, v)
                    - 2.0 * (self.price)(S + dS, K, T, r, q, v)
                    + (self.price)(S, K, T, r, q, v))
                    / (dS * dS)
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn theta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dT: f64,                  // = 1 / 365,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T - dT, r, q, v) - (self.price)(S, K, T, r, q, v)) / dT
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T - dT, r, q, v) - (self.price)(S, K, T + dT, r, q, v))
                    / (2.0 * dT)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, q, v) - (self.price)(S, K, T + dT, r, q, v)) / dT
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn vega(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dv: f64,                  // = 0.001,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, q, v) - (self.price)(S, K, T, r, q, v - dv)) / dv
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r, q, v + dv) - (self.price)(S, K, T, r, q, v - dv))
                    / (2.0 * dv)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, q, v + dv) - (self.price)(S, K, T, r, q, v)) / dv
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn rho(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dr: f64,                  // = 0.001,
        method: DifferenceMethod, //  = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r + dr, q, v) - (self.price)(S, K, T, r - dr, q, v))
                    / (2.0 * dr)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r + dr, q, v) - (self.price)(S, K, T, r, q, v)) / dr
            }
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, q, v) - (self.price)(S, K, T, r - dr, q, v)) / dr
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn carry(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dq: f64,                  // = 0.001,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, q, v) - (self.price)(S, K, T, r, q - dq, v)) / dq
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r, q + dq, v) - (self.price)(S, K, T, r, q - dq, v))
                    / (2.0 * dq)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, q + dq, v) - (self.price)(S, K, T, r, q, v)) / dq
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn elasticity(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        return ((self.price)(S + dS, K, T, r, q, v) - (self.price)(S - dS, K, T, r, q, v))
            / (2.0 * dS)
            * S
            / (self.price)(S, K, T, r, q, v);
    }

    #[allow(non_snake_case)]
    pub fn speed(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        return ((self.price)(S + 2.0 * dS, K, T, r, q, v)
            - 3.0 * (self.price)(S + dS, K, T, r, q, v)
            + 3.0 * (self.price)(S, K, T, r, q, v)
            - (self.price)(S - dS, K, T, r, q, v))
            / (dS * dS * dS);
    }

    #[allow(non_snake_case)]
    pub fn deltap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S * (1.0 + dS), K, T, r, q, v) - (self.price)(S * (1.0 - dS), K, T, r, q, v))
            * 2.0
            / S
    }

    #[allow(non_snake_case)]
    pub fn gammap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        S / 100.0
            * ((self.price)(S + dS, K, T, r, q, v) - 2.0 * (self.price)(S, K, T, r, q, v)
                + (self.price)(S - dS, K, T, r, q, v))
            / (dS * dS)
    }

    #[allow(non_snake_case)]
    pub fn vegap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dv: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S, K, T, r, q, v + dv) - (self.price)(S, K, T, r, q, v - dv)) * v / 0.1 / 2.0
    }

    /// Also known as DdeltaDvol
    #[allow(non_snake_case)]
    pub fn vanna(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01,
        dv: f64,                   // = 0.001
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S + dS, K, T, r, q, v + dv)
            - (self.price)(S + dS, K, T, r, q, v - dv)
            - (self.price)(S - dS, K, T, r, q, v + dv)
            + (self.price)(S - dS, K, T, r, q, v - dv))
            / (4.0 * dS)
            / dv
    }

    /// Also known as DdeltaDtime
    #[allow(non_snake_case)]
    pub fn charm(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01,
        dT: f64,                   // = 1 / 365
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S + dS, K, T + dT, r, q, v)
            - (self.price)(S + dS, K, T - dT, r, q, v)
            - (self.price)(S - dS, K, T + dT, r, q, v)
            + (self.price)(S - dS, K, T - dT, r, q, v))
            / (4.0 * dS)
            / -dT
    }

    #[allow(non_snake_case)]
    pub fn dgamma_dvol(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dS: f64,                   // = 0.01,
        dv: f64,                   // = 0.001
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S + dS, K, T, r, q, v + dv) - 2.0 * (self.price)(S, K, T, r, q, v + dv)
            + (self.price)(S - dS, K, T, r, q, v + dv)
            - (self.price)(S + dS, K, T, r, q, v - dv)
            + 2.0 * (self.price)(S, K, T, r, q, v - dv)
            - (self.price)(S - dS, K, T, r, q, v - dv))
            / (2.0 * dv * (dS * dS))
    }

    #[allow(non_snake_case)]
    pub fn vomma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dv: f64,                   // = 0.001,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        // DvegaDvol
        ((self.price)(S, K, T, r, q, v + dv) - 2.0 * (self.price)(S, K, T, r, q, v)
            + (self.price)(S, K, T, r, q, v - dv))
            / (dv * dv)
    }

    #[allow(non_snake_case)]
    pub fn time_gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dT: f64,                   // = 1 / 365,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        return ((self.price)(S, K, T + dT, r, q, v) - 2.0 * (self.price)(S, K, T, r, q, v)
            + (self.price)(S, K, T - dT, r, q, v))
            / (dT * dT);
    }

    #[allow(non_snake_case)]
    pub fn futures_rho(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dr: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S, K, T, r + dr, q, v) - (self.price)(S, K, T, r - dr, q, v)) / 2.0
    }

    #[allow(non_snake_case)]
    pub fn rho2(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dq: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S, K, T, r, q - dq, v) - (self.price)(S, K, T, r, q + dq, v)) / 2.0
    }

    #[allow(non_snake_case)]
    pub fn strike_delta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dK: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S, K + dK, T, r, q, v) - (self.price)(S, K - dK, T, r, q, v)) / (2.0 * dK)
    }

    #[allow(non_snake_case)]
    pub fn strike_gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        q: f64,
        v: f64,
        dK: f64,                   // = 0.01,
        _method: DifferenceMethod, // = 'central'
    ) -> f64 {
        ((self.price)(S, K + dK, T, r, q, v) - 2.0 * (self.price)(S, K, T, r, q, v)
            + (self.price)(S, K - dK, T, r, q, v))
            / (dK * dK)
    }
}
