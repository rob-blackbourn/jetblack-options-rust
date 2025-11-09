/// Class for calculating numeric greeks for options using finite difference
/// methods for the style with no carry or dividend yield.

// from typing import Callable, Literal

pub struct NumericGreeks {
    /// fn price(S: 64, K: f64, T, r: f64, v: f64) -> f64
    pub price: Box<dyn Fn(f64, f64, f64, f64, f64) -> f64>,
}

pub enum DifferenceMethod {
    Central,
    Backward,
    Forward,
}

impl NumericGreeks {
    pub fn new(price: impl Fn(f64, f64, f64, f64, f64) -> f64 + 'static) -> Self {
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
        v: f64,
        dS: Option<f64>,
        method: Option<DifferenceMethod>, // = 'central'
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        let method = method.unwrap_or(DifferenceMethod::Central);
        match method {
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, T, r, v) - (self.price)(S - dS, K, T, r, v)) / (2.0 * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + dS, K, T, r, v) - (self.price)(S, K, T, r, v)) / dS
            }
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, v) - (self.price)(S - dS, K, T, r, v)) / dS
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
        v: f64,
        dS: Option<f64>,
        method: Option<DifferenceMethod>,
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        let method = method.unwrap_or(DifferenceMethod::Central);
        match method {
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, T, r, v) - 2.0 * (self.price)(S, K, T, r, v)
                    + (self.price)(S - dS, K, T, r, v))
                    / (dS * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + 2.0 * dS, K, T, r, v) - 2.0 * (self.price)(S + dS, K, T, r, v)
                    + (self.price)(S, K, T, r, v))
                    / (dS * dS)
            }
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, v) - 2.0 * (self.price)(S - dS, K, T, r, v)
                    + (self.price)(S - 2.0 * dS, K, T, r, v))
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
        v: f64,
        dT: Option<f64>,                  // = 1.0 / 365.0,
        method: Option<DifferenceMethod>, // = 'central'
    ) -> f64 {
        let dT = dT.unwrap_or(1.0 / 365.0);
        let method = method.unwrap_or(DifferenceMethod::Central);
        match method {
            DifferenceMethod::Central => {
                ((self.price)(S, K, T - dT, r, v) - (self.price)(S, K, T + dT, r, v)) / (2.0 * dT)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, v) - (self.price)(S, K, T + dT, r, v)) / dT
            }
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T - dT, r, v) - (self.price)(S, K, T, r, v)) / dT
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
        v: f64,
        dv: Option<f64>,                  // = 0.001,
        method: Option<DifferenceMethod>, // = 'central'
    ) -> f64 {
        let dv = dv.unwrap_or(0.001);
        let method = method.unwrap_or(DifferenceMethod::Central);
        match method {
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r, v + dv) - (self.price)(S, K, T, r, v - dv)) / (2.0 * dv)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, v + dv) - (self.price)(S, K, T, r, v)) / dv
            }
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, v) - (self.price)(S, K, T, r, v - dv)) / dv
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
        v: f64,
        dr: Option<f64>,                  // = 0.001,
        method: Option<DifferenceMethod>, // = 'central'
    ) -> f64 {
        let dr = dr.unwrap_or(0.001);
        let method = method.unwrap_or(DifferenceMethod::Central);
        match method {
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r + dr, v) - (self.price)(S, K, T, r - dr, v)) / (2.0 * dr)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r + dr, v) - (self.price)(S, K, T, r, v)) / dr
            }
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, v) - (self.price)(S, K, T, r - dr, v)) / dr
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
        v: f64,
        dS: Option<f64>,                  // = 0.01,
        method: Option<DifferenceMethod>, // = 'central'
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        let method = method.unwrap_or(DifferenceMethod::Central);
        self.delta(S, K, T, r, v, Some(dS), Some(method)) * S / (self.price)(S, K, T, r, v)
    }

    #[allow(non_snake_case)]
    pub fn speed(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dS: Option<f64>, // = 0.01,
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        ((self.price)(S + 2.0 * dS, K, T, r, v) - 3.0 * (self.price)(S + dS, K, T, r, v)
            + 3.0 * (self.price)(S, K, T, r, v)
            - (self.price)(S - dS, K, T, r, v))
            / (dS * dS * dS)
    }

    #[allow(non_snake_case)]
    pub fn deltap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dS: Option<f64>, // = 0.01,
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        ((self.price)(S * (1.0 + dS), K, T, r, v) - (self.price)(S * (1.0 - dS), K, T, r, v))
            / (2.0 * S)
    }

    #[allow(non_snake_case)]
    pub fn gammap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dS: Option<f64>,                  // = 0.01,
        method: Option<DifferenceMethod>, // = 'central'
    ) -> f64 {
        S / 100.0 * self.gamma(S, K, T, r, v, dS, method)
    }

    #[allow(non_snake_case)]
    pub fn vegap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dv: Option<f64>, // = 0.01,
    ) -> f64 {
        let dv = dv.unwrap_or(0.01);
        ((self.price)(S, K, T, r, v + dv) - (self.price)(S, K, T, r, v - dv)) * v / 0.1 / 2.0
    }

    #[allow(non_snake_case)]
    pub fn vanna(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dS: Option<f64>, // = 0.01,
        dv: Option<f64>, // = 0.001
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        let dv = dv.unwrap_or(0.001);
        // Also known as DdeltaDvol
        ((self.price)(S + dS, K, T, r, v + dv)
            - (self.price)(S + dS, K, T, r, v - dv)
            - (self.price)(S - dS, K, T, r, v + dv)
            + (self.price)(S - dS, K, T, r, v - dv))
            / (4.0 * dS)
            / dv
    }

    #[allow(non_snake_case)]
    pub fn charm(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dS: Option<f64>, // = 0.01,
        dT: Option<f64>, // = 1 / 365
    ) -> f64 {
        // Also known as DdeltaDtime
        let dS = dS.unwrap_or(0.01);
        let dT = dT.unwrap_or(1.0 / 365.0);
        ((self.price)(S + dS, K, T + dT, r, v)
            - (self.price)(S + dS, K, T - dT, r, v)
            - (self.price)(S - dS, K, T + dT, r, v)
            + (self.price)(S - dS, K, T - dT, r, v))
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
        v: f64,
        dS: Option<f64>, // = 0.01,
        dv: Option<f64>, // = 0.01
    ) -> f64 {
        let dS = dS.unwrap_or(0.01);
        let dv = dv.unwrap_or(0.01);
        ((self.price)(S + dS, K, T, r, v + dv) - 2.0 * (self.price)(S, K, T, r, v + dv)
            + (self.price)(S - dS, K, T, r, v + dv)
            - (self.price)(S + dS, K, T, r, v - dv)
            + 2.0 * (self.price)(S, K, T, r, v - dv)
            - (self.price)(S - dS, K, T, r, v - dv))
            / (2.0 * dv * (dS * dS))
            / 100.0
    }

    #[allow(non_snake_case)]
    pub fn vomma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dv: Option<f64>, // = 0.001,
    ) -> f64 {
        // DvegaDvol
        let dv = dv.unwrap_or(0.001);
        ((self.price)(S, K, T, r, v + dv) - 2.0 * (self.price)(S, K, T, r, v)
            + (self.price)(S, K, T, r, v - dv))
            / (dv * dv)
    }

    #[allow(non_snake_case)]
    pub fn time_gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dT: Option<f64>, // = 1 / 365,
    ) -> f64 {
        let dT = dT.unwrap_or(1.0 / 365.0);
        ((self.price)(S, K, T + dT, r, v) - 2.0 * (self.price)(S, K, T, r, v)
            + (self.price)(S, K, T - dT, r, v))
            / (dT * dT)
    }

    #[allow(non_snake_case)]
    pub fn strike_delta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dX: Option<f64>, // = 0.01,
    ) -> f64 {
        let dX = dX.unwrap_or(0.01);
        ((self.price)(S, K + dX, T, r, v) - (self.price)(S, K - dX, T, r, v)) / (2.0 * dX)
    }

    #[allow(non_snake_case)]
    pub fn strike_gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        v: f64,
        dX: Option<f64>, // = 0.01,
    ) -> f64 {
        let dX = dX.unwrap_or(0.01);
        ((self.price)(S, K + dX, T, r, v) - 2.0 * (self.price)(S, K, T, r, v)
            + (self.price)(S, K - dX, T, r, v))
            / (dX * dX)
    }
}
