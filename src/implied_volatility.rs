use libm::fabs;

const MAX_ITERATIONS: i32 = 20;
const EPSILON: f64 = 1e-8;

/// Calculate the volatility of an option that is implied by the price.
///
/// Args:
///     p (f64): The price.
///     price: A function to calculate the price given a volatility.
///     max_iterations (Option<i32>): The maximum number of iterations. Defaults to 20.
///     epsilon (Option<f64>): The maximum error. Defaults to 1e8.
///
/// Result:
///     f64: The implied volatility.
pub fn solve_ivol(
    p: f64,
    price: impl Fn(f64) -> f64,
    max_iterations: Option<i32>,
    epsilon: Option<f64>,
) -> f64 {
    let max_iterations = max_iterations.unwrap_or(MAX_ITERATIONS);
    let epsilon = epsilon.unwrap_or(EPSILON);

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
