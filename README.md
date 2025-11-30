# jetblack-options-rust

A rust library for option calculations.

## Status

This is work in progress.

## Overview

This library provides two things:

- Valuation functions for options and other volatility products
- A bumping framework to calculate sensitivities from prices

## Examples

Some calculations with Black-Scholes-Merton.

```rust
use jetblack_options::european::BlackScholesMerton;

#[allow(non_snake_case)]
fn main() {
    // The optional calculator inputs.
    let is_call = true; // It's a call.
    let S = 100.0; // The stock price is 100.
    let K = 101.0; // The strike price is 101.
    let t = 30.0 / 365.0; // The time to expiry is in years (30 days).
    let r = 3.0 / 100.0; // The risk free rate is 3%.
    let q = 6.0 / 100.0; // The dividend yield is 6%.
    let v = 15.0 / 100.0; // The volatility is 15%.

    // Price the option.
    let p = BlackScholesMerton::price(is_call, S, K, t, r, q, v);
    println!("The price is {}", p);

    // The delta.
    let d = BlackScholesMerton::delta(is_call, S, K, t, r, q, v);
    println!("The delta is {}", d);

    // The theta should predict tomorrows price.
    let t = BlackScholesMerton::theta(is_call, S, K, t, r, q, v);
    println!("The theta is {} (29 day price is {}", t, p + t/365.0);

    let T1 = 29.0 / 365.0; // The time to expiry in years (29 days).
    let p1 = BlackScholesMerton::price(is_call, S, K, T1, r, q, v);
    println!("The price at 29 days is {}", p1);
}
```

The European binomial tree model only provides a price. The greeks
will be calculated using finite difference methods (bumping).

```rust
use jetblack_options::trees::EuropeanBinomial;
use jetblack_options::fdm::DifferenceMethod;

#[allow(non_snake_case)]
fn main() {
    // The optional calculator inputs.
    let is_call = true; // It's a call.
    let S = 100.0; // The stock price is 100.
    let K = 101.0; // The strike price is 101.
    let T = 30.0 / 365.0; // The time to expiry is in years (30 days).
    let r = 3.0 / 100.0; // The risk free rate is 3%.
    let q = 6.0 / 100.0; // The dividend yield is 6%.
    let b = r - q; // The cost of carry.
    let v = 15.0 / 100.0; // The volatility is 15%.
    let n = 100; // The number of steps.

    // Price the option.
    let p = EuropeanBinomial::price(is_call, S, K, T, r, b, v, n);
    println!("The price is {}", p);

    // Create the bumper.
    let bumper = EuropeanBinomial::fdm_greeks(is_call, n);

    // The delta
    let d1 = bumper.delta(S, K, T, r, b, v, 0.0001, DifferenceMethod::Central);
    println!("The bumped delta is {}", d1);

    // The theta should predict tomorrows price.
    let t = bumper.theta(S, K, T, r, b, v, 1.0/365.0/60.0, DifferenceMethod::Central);
    println!("The theta is {} (29 day price is {}", t, p + t/365.0);

    let t1 = 29.0 / 365.0; // The time to expiry in years (29 days).
    let p1 = EuropeanBinomial::price(is_call, S, K, t1, r, b, v, n);
    println!("The price at 29 days is {}", p1);
}
```

## Coverage

### European

* Black 76
* Black-Scholes 73
* Black-Scholes-Merton
* Garman Kohlhagen
* Generalize Black-Scholes

### American

* Barone, Adesi and Whaley (1987)
* Bjerksund and Stensland (1993)
* Bjerksund and Stensland (2002)

### Trees

* Cox, Ross & Rubinstein
* European Binomial
* Jarrow-Rudd
* Leisen Reimer
* Trinomial

### Bumping (Finite Difference Methods)

* With Carry
* Without Carry
* With Dividend Yield
