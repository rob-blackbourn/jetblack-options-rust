# jetblack-options-rust

A rust library for option calculations.

## Status

This is work in progress.

## Overview

This library provides two things:

- Valuation functions for options and other volatility products
- A bumping framework to calculate sensitivities from prices

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
