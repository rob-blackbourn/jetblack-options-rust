//! Option Valuations
//!
//! This create provides:
//!
//! * Option pricers
//! * Finite different method functions for calculating greeks.

mod distributions;
mod implied_volatility;

pub mod american;
pub mod european;
pub mod fdm;
pub mod trees;
pub mod variance;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
