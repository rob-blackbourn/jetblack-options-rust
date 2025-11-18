//! Approximation of inverse chi-square (Weiss and Greenhall, 1996)
//! Restrictions: df >= 1 and 0.005 <= p <= 0.995
//! df can in theory be non-integral, but we define it as int here.
//! Returns, given the desired p-value and a degree of freedom df,
//! the corresponding chi-square value.
//! Max. error of approximation is 3%
//! See: https://apps.dtic.mil/sti/pdfs/ADA515532.pdf

use libm::{exp, fmin, log, pow, sqrt};

#[allow(non_snake_case)]
pub fn chi_inv(p: f64, df: i32) -> f64 {
    let p = 1.0 - p;
    if p <= 0.5 && df <= 10 {
        let c1 = -0.5748646;
        let c2 = 0.9512363;
        let c3 = -0.6998588;
        let c4 = 0.4245549;
        let c5 = -0.1010678;

        let a = (df as f64) / 2.0;
        let n = a as i32;
        let y = a - (n as f64);
        let mut G = 1.0 + y * (c1 + y * (c2 + y * (c3 + y * (c4 + y * c5))));
        for k in 1..n + 1 {
            G = G * (y + (k as f64));
        }
        let A = p * G;
        let mut u = 0.0;
        for _ in 0..8 {
            let g = 1.0 + (u / (a + 1.0)) * (1.0 + (u / (a + 2.0)) * (1.0 + (u / (a + 3.0))));
            u = pow(A * exp(u) / g, 1.0 / a);
        }
        let x = 2.0 * u;
        x
    } else {
        let a0 = 2.30753;
        let a1 = 0.27601;
        let b1 = 0.99229;
        let b2 = 0.04481;

        let p1 = fmin(p, 1.0 - p);
        let t = sqrt(-2.0 * log(p1));
        let X = t - (a0 + a1 * t) / (1.0 + b1 * t + b2 * t * t);
        let s = if p - 0.5 < 0.0 {
            -1
        } else if p - 0.5 > 0.0 {
            1
        } else {
            0
        };
        let df = df as f64;
        let b = 2.0 / (9.0 * df);
        let x = df * pow((1.0 - b + (s as f64) * X * sqrt(b)), 3.0);
        x
    }
}

#[cfg(test)]
mod tests {
    use libm::fabs;

    use super::*;

    fn is_close_to(actual: f64, expected: f64, threshold: f64) -> bool {
        let diff = fabs(actual - expected);
        diff < threshold
    }

    #[test]
    fn it_should_calc_chi_inv() {
        let actual = chi_inv(0.1, 5);
        let expected = 9.184520236568392;
        assert!(is_close_to(actual, expected, 1e-12));

        let actual = chi_inv(0.1, 10);
        let expected = 15.937814138474298;
        assert!(is_close_to(actual, expected, 1e-12));

        let actual = chi_inv(0.7, 5);
        let expected = 3.0123268283837;
        assert!(is_close_to(actual, expected, 1e-12));

        let actual = chi_inv(0.7, 10);
        let expected = 7.356372548526449;
        assert!(is_close_to(actual, expected, 1e-12));
    }
}
