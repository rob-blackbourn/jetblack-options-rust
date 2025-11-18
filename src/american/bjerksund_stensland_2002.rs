//! # Option valuation functions implementing the Bjerksund and Stensland (2002)
//!
//! American approximation

use libm::{exp, fmax, log, pow, sqrt};

use crate::european::generalised_black_scholes::price as bs_price;
use crate::{implied_volatility::solve_ivol, numeric_greeks::with_carry::NumericGreeks};

fn cdf(x: f64) -> f64 {
    crate::distributions::cdf(x, 0.0, 1.0)
}

fn sqr(x: f64) -> f64 {
    x * x
}

#[allow(non_snake_case)]
fn _phi(S: f64, T: f64, gamma_: f64, h: f64, i: f64, r: f64, b: f64, v: f64) -> f64 {
    let lambda_ = (-r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1.0) * (v * v)) * T;
    let d = -(log(S / h) + (b + (gamma_ - 0.5) * (v * v)) * T) / (v * sqrt(T));
    let kappa = 2.0 * b / (v * v) + 2.0 * gamma_ - 1.0;
    exp(lambda_)
        * pow(S, gamma_)
        * (cdf(d) - pow((i / S), kappa) * cdf(d - 2.0 * log(i / S) / (v * sqrt(T))))
}

// #[allow(non_snake_case)]
// fn _ksi(
//     S: f64,
//     T2: f64,
//     gamma_: f64,
//     h: f64,
//     I2: f64,
//     I1: f64,
//     t1: f64,
//     r: f64,
//     b: f64,
//     v: f64,
// ) -> f64 {
//     let e1 = (log(S / I1) + (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));
//     let e2 = (log((I2 * I2) / (S * I1)) + (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));
//     let e3 = (log(S / I1) - (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));
//     let e4 = (log(I2 * I2 / (S * I1)) - (b + (gamma_ - 0.5) * (v * v)) * t1) / (v * sqrt(t1));

//     let f1 = (log(S / h) + (b + (gamma_ - 0.5) * (v * v)) * T2) / (v * sqrt(T2));
//     let f2 = (log((I2 * I2) / (S * h)) + (b + (gamma_ - 0.5) * (v * v)) * T2) / (v * sqrt(T2));
//     let f3 = (log((I1 * I1) / (S * h)) + (b + (gamma_ - 0.5) * (v * v)) * T2) / (v * sqrt(T2));
//     let f4 = (log(S * (I1 * I1) / (h * (I2 * I2))) + (b + (gamma_ - 0.5) * (v * v)) * T2)
//         / (v * sqrt(T2));

//     let rho = sqrt(t1 / T2);
//     let lambda_ = -r + gamma_ * b + 0.5 * gamma_ * (gamma_ - 1) * (v * v);
//     let kappa = 2.0 * b / (v * v) + (2.0 * gamma_ - 1);

//     exp(lambda_ * T2)
//         * pow(S, gamma_)
//         * (cbnd(-e1, -f1, rho)
//             - (I2 / S) * *kappa * cbnd(-e2, -f2, rho)
//             - (I1 / S) * *kappa * cbnd(-e3, -f3, -rho)
//             + (I1 / I2) * *kappa * cbnd(-e4, -f4, -rho))
// }
