use libm::fabs;

/// Calculate the volatility of an option that is implied by the price.
///
/// Args:
///     p (f64): The price.
///     price: A function to calculate the price given a volatility.
///     max_iterations (usize): The maximum number of iterations. A typical value is 20.
///     epsilon (f64): The maximum error. A typical value is 1e8.
///
/// Result:
///     f64: The implied volatility.
pub fn solve_ivol(
    p: f64,
    price: impl Fn(f64) -> f64,
    max_iterations: usize, // TODO: Should be usize?
    epsilon: f64,
) -> f64 {
    let mut v_lo = 0.005;
    let mut v_hi = 4.0;
    let mut p_lo = price(v_lo);
    let mut p_hi = price(v_hi);

    let mut n = 0;
    let mut v = v_lo + (p - p_lo) * (v_hi - v_lo) / (p_hi - p_lo);
    let mut p1 = price(v);
    while fabs(p - p1) > epsilon && n < max_iterations {
        n += 1;

        if p1 < p {
            v_lo = v;
        } else {
            v_hi = v;
        }

        p_lo = price(v_lo);
        p_hi = price(v_hi);
        v = v_lo + (p - p_lo) * (v_hi - v_lo) / (p_hi - p_lo);
        p1 = price(v);
    }

    return v;
}
