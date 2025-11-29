//! Calculating numeric greeks for options using finite difference methods for pricers using using cost of carry.

use crate::fdm::DifferenceMethod;

/// A struct for calculating option sensitivities using finite difference methods for pricers using cost of carry.
pub struct FdmWithCarry {
    /// A function to calculate the price of an option.
    ///
    /// fn price(S: 64, K: f64, T, r: f64, b: f64, v: f64) -> f64
    pub price: Box<dyn Fn(f64, f64, f64, f64, f64, f64) -> f64>,
}

impl FdmWithCarry {
    /// Create a finite difference calculator given a pricing function.
    pub fn new(price: impl Fn(f64, f64, f64, f64, f64, f64) -> f64 + 'static) -> Self {
        FdmWithCarry {
            price: Box::new(price),
        }
    }

    /// Calculate the delta on an option using the finite difference.
    ///
    /// The delta is calculated according to one of the three difference methods.
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial S} = \frac{BS_{price}(S, K, T, r, b, \sigma) - BS_{price}(S - \Delta S, K, T, r, b, \sigma)}{\Delta S}
    /// $$
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial S} = \frac{BS_{price}(S + \Delta S, K, T, r, b, \sigma) - BS_{price}(S-\Delta S, K, T, r, b, \sigma)}{2 \Delta S}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial S} = \frac{BS_{price}(S+\Delta S, K, T, r, b, \sigma) - BS_{price}(S, K, T, r, b, \sigma)}{\Delta S}
    /// $$
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * dS (f64): The absolute amount to change the asset price by. A common value is0.01.
    /// * method (DifferenceMethod): The method to use. A common value is 'central'.
    ///
    /// ### Returns
    ///
    /// f64: The numeric delta.
    #[allow(non_snake_case)]
    pub fn delta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dS: f64,
        method: DifferenceMethod,
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                return ((self.price)(S, K, T, r, b, v) - (self.price)(S - dS, K, T, r, b, v)) / dS;
            }
            DifferenceMethod::Central => {
                return ((self.price)(S + dS, K, T, r, b, v) - (self.price)(S - dS, K, T, r, b, v))
                    / (2.0 * dS);
            }
            DifferenceMethod::Forward => {
                return ((self.price)(S + dS, K, T, r, b, v) - (self.price)(S, K, T, r, b, v)) / dS;
            }
        }
    }

    /// Calculate the gamma of an option using finite difference methods.
    ///
    /// The gamma is calculated according to one of the three difference methods.
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial^2 V}{\partial S^2} = \frac{BS_{price}(S, K, T, r, b, \sigma) - 2 BS_{price}(S - \Delta S, K, T, r, b, \sigma) + BS_{price}(S - 2 \Delta S, K, T, r, b, \sigma)}{\Delta S^2}
    /// $$
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial^2 V}{\partial S^2} = \frac{BS_{price}(S + \Delta S, K, T, r, b, \sigma) - 2 BS_{price}(S, K, T, r, b, \sigma) + BS_{price}(S - \Delta S, K, T, r, b, \sigma)}{\Delta S^2}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial^2 V}{\partial S^2} = \frac{BS_{price}(S + 2 \Delta S, K, T, r, b, \sigma) - 2 BS_{price}(S + \Delta S, K, T, r, b, \sigma) + BS_{price}(S, K, T, r, b, \sigma)}{\Delta S^2}
    /// $$
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * dS (f64): The absolute amount to change the asset price by. A common value is 0.01.
    /// * method (DifferenceMethod): The method to use. A common value is central.
    ///
    /// ### Returns
    ///
    /// f64: The numeric gamma.
    #[allow(non_snake_case)]
    pub fn gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dS: f64,                  // = 0.01,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, b, v) - 2.0 * (self.price)(S - dS, K, T, r, b, v)
                    + (self.price)(S - 2.0 * dS, K, T, r, b, v))
                    / (dS * dS)
            }
            DifferenceMethod::Central => {
                ((self.price)(S + dS, K, T, r, b, v) - 2.0 * (self.price)(S, K, T, r, b, v)
                    + (self.price)(S - dS, K, T, r, b, v))
                    / (dS * dS)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S + 2.0 * dS, K, T, r, b, v)
                    - 2.0 * (self.price)(S + dS, K, T, r, b, v)
                    + (self.price)(S, K, T, r, b, v))
                    / (dS * dS)
            }
        }
    }

    /// Calculate the theta on an option using the finite difference.
    ///
    /// The theta is calculated according to one of the three difference methods.
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial T} = \frac{BS_{price}(S, K, T - \Delta T, r, b, \sigma) - BS_{price}(S, K, T, r, b, \sigma)}{\Delta T}
    /// $$
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial T} = \frac{BS_{price}(S, K, T - \Delta T, r, b, \sigma) - BS_{price}(S, K, T + \Delta T, r, b, \sigma)}{2 \Delta T}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial T} = \frac{BS_{price}(S, K, T, r, b, \sigma) - BS_{price}(S, K, T + \Delta T, r, b, \sigma)}{\Delta T}
    /// $$
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * dT (f64): The absolute amount to change the asset price by. A common value is 1/365.
    /// * method (DifferenceMethod): The method to use. A common value is central.
    ///
    /// ### Returns
    ///
    /// f64: The numeric theta.
    #[allow(non_snake_case)]
    pub fn theta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dT: f64,                  // = 1 / 365,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T - dT, r, b, v) - (self.price)(S, K, T, r, b, v)) / dT
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T - dT, r, b, v) - (self.price)(S, K, T + dT, r, b, v))
                    / (2.0 * dT)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, b, v) - (self.price)(S, K, T + dT, r, b, v)) / dT
            }
        }
    }

    /// Calculate the vega on an option using the finite difference.
    ///
    /// The vega is calculated according to one of the three difference methods.
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial \sigma} = \frac{BS_{price}(S, K, T, r, b, \sigma) - BS_{price}(S, K, T, r, b, \sigma - \Delta \sigma)}{\Delta \sigma}
    /// $$
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial \sigma} = \frac{BS_{price}(S, K, T, r, b, \sigma + \Delta \sigma) - BS_{price}(S, K, T, r, b, \sigma - \Delta \sigma)}{2 \Delta \sigma}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial \sigma} = \frac{BS_{price}(S, K, T, r, b, \sigma + \Delta \sigma) - BS_{price}(S, K, T, r, b, \sigma)}{\Delta \sigma}
    /// $$
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * dV (f64): The absolute amount to change the volatility by. A common value is 0.001.
    /// * method (DifferenceMethod): The method to use. A common value is central.
    ///
    /// ### Returns
    ///
    /// f64: The numeric vega.
    #[allow(non_snake_case)]
    pub fn vega(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dv: f64,                  // = 0.001,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, b, v) - (self.price)(S, K, T, r, b, v - dv)) / dv
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r, b, v + dv) - (self.price)(S, K, T, r, b, v - dv))
                    / (2.0 * dv)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, b, v + dv) - (self.price)(S, K, T, r, b, v)) / dv
            }
        }
    }

    /// Calculate the rho on an option using the finite difference.
    ///
    /// The rho is calculated according to one of the three difference methods.
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial r} = \frac{BS_{price}(S, K, T, r, b, \sigma) - BS_{price}(S, K, T, r - \Delta r, b - \Delta r, \sigma)}{\Delta r}
    /// $$
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial r} = \frac{BS_{price}(S, K, T, r + \Delta r, b + \Delta r, \sigma) - BS_{price}(S, K, T, r - \Delta r, b - \Delta r, \sigma)}{2 \Delta r}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial r} = \frac{BS_{price}(S, K, T, r + \Delta r, b + \Delta r, \sigma) - BS_{price}(S, K, T, r, b, \sigma)}{\Delta r}
    /// $$
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * dr (f64): The absolute amount to change the rate by. A common value is 0.001.
    /// * method (DifferenceMethod): The method to use. A common value is 'central'.
    ///
    /// ### Returns
    ///
    /// f64: The numeric rho.
    #[allow(non_snake_case)]
    pub fn rho(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dr: f64,                  // = 0.001,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r + dr, b, v) - (self.price)(S, K, T, r - dr, b - dr, v))
                    / dr
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r + dr, b + dr, v)
                    - (self.price)(S, K, T, r - dr, b - dr, v))
                    / (2.0 * dr)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r + dr, b + dr, v) - (self.price)(S, K, T, r - dr, b, v))
                    / dr
            }
        }
    }

    /// Calculate the carry on an option using the finite difference.
    ///
    /// The carry is calculated according to one of the three difference methods.
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial r} = \frac{BS_{price}(S, K, T, r, b, \sigma) - BS_{price}(S, K, T, r, b - \Delta b, \sigma)}{\Delta b}
    /// $$
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial b} = \frac{BS_{price}(S, K, T, r, b + \Delta b, \sigma) - BS_{price}(S, K, T, r, b - \Delta b, \sigma)}{2 \Delta b}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial V}{\partial b} = \frac{BS_{price}(S, K, T, r, b + \Delta b, \sigma) - BS_{price}(S, K, T, r, b, \sigma)}{\Delta b}
    /// $$
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * db (f64): The absolute amount to change the carry rate by. A common value is 0.001.
    /// * method (DifferenceMethod): The method to use. A common value is 'central'.
    ///
    /// ### Returns
    ///
    /// f64: The numeric carry.
    #[allow(non_snake_case)]
    pub fn carry(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        db: f64,                  // = 0.001,
        method: DifferenceMethod, // = 'central'
    ) -> f64 {
        match method {
            DifferenceMethod::Backward => {
                ((self.price)(S, K, T, r, b, v) - (self.price)(S, K, T, r, b - db, v)) / db
            }
            DifferenceMethod::Central => {
                ((self.price)(S, K, T, r, b + db, v) - (self.price)(S, K, T, r, b - db, v))
                    / (2.0 * db)
            }
            DifferenceMethod::Forward => {
                ((self.price)(S, K, T, r, b + db, v) - (self.price)(S, K, T, r, b, v)) / db
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
        b: f64,
        v: f64,
        dS: f64,
        method: DifferenceMethod,
    ) -> f64 {
        self.delta(S, K, T, r, b, v, dS, method) * S / (self.price)(S, K, T, r, b, v)
    }

    #[allow(non_snake_case)]
    pub fn speed(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dS: f64, // = 0.01,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S + 2.0 * dS, K, T, r, b, v) - 3.0 * (self.price)(S + dS, K, T, r, b, v)
            + 3.0 * (self.price)(S, K, T, r, b, v)
            - (self.price)(S - dS, K, T, r, b, v))
            / (dS * dS * dS)
    }

    #[allow(non_snake_case)]
    pub fn deltap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dS: f64, // = 0.01
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S * (1.0 + dS), K, T, r, b, v) - (self.price)(S * (1.0 - dS), K, T, r, b, v))
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
        b: f64,
        v: f64,
        dS: f64, // = 0.01,
        method: DifferenceMethod,
    ) -> f64 {
        self.gamma(S, K, T, r, b, v, dS, method) * S / 100.0
    }

    #[allow(non_snake_case)]
    pub fn vegap(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dv: f64, // = 0.001,
        method: DifferenceMethod,
    ) -> f64 {
        self.vega(S, K, T, r, b, v, dv, method) * v * 10.0
    }

    /// The second order derivative of the option price to a change in the asset
    /// price and a change in the volatility.
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike price.
    /// * T (f64): The time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The asset volatility.
    /// * dS (f64): The change in spot price. A common value is 0.01.
    /// * dv (f64): The change in volatility. A common value is 0.01.
    ///
    /// ### Returns
    ///
    /// f64: The vanna
    #[allow(non_snake_case)]
    pub fn vanna(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dS: f64, // = 0.01,
        dv: f64, // = 0.001
        _method: DifferenceMethod,
    ) -> f64 {
        // Also known as DdeltaDvol
        ((self.price)(S + dS, K, T, r, b, v + dv)
            - (self.price)(S + dS, K, T, r, b, v - dv)
            - (self.price)(S - dS, K, T, r, b, v + dv)
            + (self.price)(S - dS, K, T, r, b, v - dv))
            / (4.0 * dS)
            / dv
    }

    /// Measures the instantaneous rate of change of delta over the passage of
    /// time.
    ///
    /// Also known as DdeltaDtime.
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike price.
    /// * T (f64): The time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The asset volatility.
    /// * dS (f64): Change in asset price. A common value is 0.01.
    /// * dT (f64): Change in time. A common value is 1/365.
    ///
    /// ### Returns
    ///
    /// f64: The charm.
    #[allow(non_snake_case)]
    pub fn charm(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dS: f64, // = 0.01,
        dT: f64, // = 1 / 365
        _method: DifferenceMethod,
    ) -> f64 {
        // Also known as DdeltaDtime
        ((self.price)(S + dS, K, T + dT, r, b, v)
            - (self.price)(S + dS, K, T - dT, r, b, v)
            - (self.price)(S - dS, K, T + dT, r, b, v)
            + (self.price)(S - dS, K, T - dT, r, b, v))
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
        b: f64,
        v: f64,
        dS: f64, // = 0.01,
        dv: f64, // = 0.001
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S + dS, K, T, r, b, v + dv) - 2.0 * (self.price)(S, K, T, r, b, v + dv)
            + (self.price)(S - dS, K, T, r, b, v + dv)
            - (self.price)(S + dS, K, T, r, b, v - dv)
            + 2.0 * (self.price)(S, K, T, r, b, v - dv)
            - (self.price)(S - dS, K, T, r, b, v - dv))
            / (2.0 * dv * (dS * dS))
    }

    /// Calculate the vomma of an option using finite difference methods.
    ///
    /// Also known as DvegaDvol.
    ///
    /// The vomma is calculated according to one of the three difference methods.
    ///
    /// Central difference method.
    ///
    /// $$
    /// \frac{\partial^2 V}{\partial \sigma^2} = \frac{BS_{price}(S, K, T, r, b, \sigma + \Delta \sigma) - 2 BS_{price}(S, K, T, r, b, \sigma) + BS_{price}(S, K, T, r, b, \sigma - \Delta \sigma)}{\Delta \sigma^2}
    /// $$
    ///
    /// Forward difference method.
    ///
    /// $$
    /// \frac{\partial^2 V}{\partial \sigma^2} = \frac{BS_{price}(S, K, T, r, b, \sigma + 2 \Delta \sigma) - 2 BS_{price}(S, K, T, r, b, \sigma + \Delta \sigma) + BS_{price}(S, K, T, r, b, \sigma)}{\Delta \sigma^2}
    /// $$
    ///
    /// Backward difference method.
    ///
    /// $$
    /// \frac{\partial^2 V}{\partial \sigma^2} = \frac{BS_{price}(S, K, T, r, b, \sigma) - 2 BS_{price}(S, K, T, r, b, \sigma - \Delta \sigma) + BS_{price}(S, K, T, r, b, \sigma - 2 \Delta \sigma)}{\Delta \sigma^2}
    /// $$
    ///
    ///
    /// ### Arguments
    ///
    /// * S (f64): The asset price.
    /// * K (f64): The strike.
    /// * T (f64): Time to expiry in years.
    /// * r (f64): The risk free rate.
    /// * b (f64): The cost of carry.
    /// * v (f64): The volatility.
    /// * dv (f64): The absolute amount to change the volatility price by. A common value is 0.001.
    /// * method (DifferenceMethod): The method to use. A common value is 'central'.
    ///
    /// ### Returns
    ///
    /// f64: The numeric vomma.
    #[allow(non_snake_case)]
    pub fn vomma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dv: f64, // = 0.001,
        _method: DifferenceMethod,
    ) -> f64 {
        return ((self.price)(S, K, T, r, b, v + dv) - 2.0 * (self.price)(S, K, T, r, b, v)
            + (self.price)(S, K, T, r, b, v - dv))
            / (dv * dv);
    }

    #[allow(non_snake_case)]
    pub fn time_gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dT: f64, //  = 1 / 365,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K, T + dT, r, b, v) - 2.0 * (self.price)(S, K, T, r, b, v)
            + (self.price)(S, K, T - dT, r, b, v))
            / (dT * dT)
    }

    #[allow(non_snake_case)]
    pub fn futures_rho(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dr: f64, // = 0.01,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K, T, r + dr, b, v) - (self.price)(S, K, T, r - dr, b, v)) / 2.0
    }

    #[allow(non_snake_case)]
    pub fn rho2(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        db: f64, // = 0.01,
        _method: DifferenceMethod,
    ) -> f64 {
        ((self.price)(S, K, T, r, b - db, v) - (self.price)(S, K, T, r, b + db, v)) / 2.0
    }

    #[allow(non_snake_case)]
    pub fn strike_delta(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dK: f64, // = 0.01,
    ) -> f64 {
        ((self.price)(S, K + dK, T, r, b, v) - (self.price)(S, K - dK, T, r, b, v)) / (2.0 * dK)
    }

    #[allow(non_snake_case)]
    pub fn strike_gamma(
        &self,
        S: f64,
        K: f64,
        T: f64,
        r: f64,
        b: f64,
        v: f64,
        dK: f64, // = 0.01,
    ) -> f64 {
        ((self.price)(S, K + dK, T, r, b, v) - 2.0 * (self.price)(S, K, T, r, b, v)
            + (self.price)(S, K - dK, T, r, b, v))
            / (dK * dK)
    }
}
