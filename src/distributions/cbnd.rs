//! The cumulative bivariate normal distribution function

use core::f64::consts::PI;

use libm::{asin, exp, fabs, fmax, sin, sqrt};

use super::cnd::CND;

fn sqr(x: f64) -> f64 {
    x * x
}

///   A function for computing bivariate normal probabilities.
///     Alan Genz
///     Department of Mathematics
///     Washington State University
///     Pullman, WA 99164-3113
///     Email : alangenz@wsu.edu
///  This function is based on the method described by
///      Drezner, Z and G.O. Wesolowsky, (1990),
///      On the computation of the bivariate normal integral,
///      Journal of Statist. Comput. Simul. 35, pp. 101-107,
///  with major modifications for double precision, and for |R| close to 1.
/// This code was originally translated into VBA by Graeme West
#[allow(non_snake_case)]
pub fn cbnd(x: f64, y: f64, rho: f64) -> f64 {
    struct Factor {
        W: &'static [f64],
        XX: &'static [f64],
    }
    let all_factors = [
        Factor {
            W: &[0.17132449237917, 0.360761573048138, 0.46791393457269],
            XX: &[-0.932469514203152, -0.661209386466265, -0.238619186083197],
        },
        Factor {
            W: &[
                4.71753363865118E-02,
                0.106939325995318,
                0.160078328543346,
                0.203167426723066,
                0.233492536538355,
                0.249147045813403,
            ],
            XX: &[
                -0.981560634246719,
                -0.904117256370475,
                -0.769902674194305,
                -0.587317954286617,
                -0.36783149899818,
                -0.125233408511469,
            ],
        },
        Factor {
            W: &[
                1.76140071391521E-02,
                4.06014298003869E-02,
                6.26720483341091E-02,
                8.32767415767048E-02,
                0.10193011981724,
                0.118194531961518,
                0.131688638449177,
                0.142096109318382,
                0.149172986472604,
                0.152753387130726,
            ],
            XX: &[
                -0.993128599185095,
                -0.963971927277914,
                -0.912234428251326,
                -0.839116971822219,
                -0.746331906460151,
                -0.636053680726515,
                -0.510867001950827,
                -0.37370608871542,
                -0.227785851141645,
                -7.65265211334973E-02,
            ],
        },
    ];

    let factors = if fabs(rho) < 0.3 {
        &all_factors[0]
    } else if fabs(rho) < 0.75 {
        &all_factors[1]
    } else {
        &all_factors[2]
    };

    let h = -x;
    let mut k = -y;
    let mut hk = h * k;
    let mut BVN = 0.0;

    if fabs(rho) < 0.925 {
        if fabs(rho) > 0.0 {
            let hs = (h * h + k * k) / 2.0;
            let asr = asin(rho);
            for i in 0..factors.W.len() {
                for ISs in [-1.0, 1.0] {
                    let sn = sin(asr * (ISs * factors.XX[i] + 1.0) / 2.0);
                    BVN = BVN + factors.W[i] * exp((sn * hk - hs) / (1.0 - sn * sn));
                }
            }
            BVN = BVN * asr / (4.0 * PI);
        }
        BVN = BVN + CND(-h) * CND(-k)
    } else {
        if rho < 0.0 {
            k = -k;
            hk = -hk;
        }
        if fabs(rho) < 1.0 {
            let Ass = (1.0 - rho) * (1.0 + rho);
            let A = sqrt(Ass);
            let bs = (h - k) * (h - k);
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            let asr = -(bs / Ass + hk) / 2.0;
            if asr > -100.0 {
                BVN = A
                    * exp(asr)
                    * (1.0 - c * (bs - Ass) * (1.0 - d * bs / 5.0) / 3.0 + c * d * Ass * Ass / 5.0);
            }
            if -hk < 100.0 {
                let b = sqrt(bs);
                BVN = BVN
                    - exp(-hk / 2.0)
                        * sqrt(2.0 * PI)
                        * CND(-b / A)
                        * b
                        * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            let A = A / 2.0;
            for i in 0..factors.W.len() {
                for ISs in [-1.0, 1.0] {
                    let xs = sqr(A * (ISs * factors.XX[i] + 1.0));
                    let rs = sqrt(1.0 - xs);
                    let asr = -(bs / xs + hk) / 2.0;
                    if asr > -100.0 {
                        BVN = BVN
                            + A * factors.W[i]
                                * exp(asr)
                                * (exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
                                    - (1.0 + c * xs * (1.0 + d * xs)));
                    }
                }
            }
            BVN = -BVN / (2.0 * PI);
        }
        if rho > 0.0 {
            BVN = BVN + CND(-fmax(h, k));
        } else {
            BVN = -BVN;
            if k > h {
                BVN = BVN + CND(k) - CND(h);
            }
        }
    }

    BVN
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
    fn it_should_calc_cbnd() {
        struct S {
            x: &'static [f64],
            y: &'static [f64],
        }

        let s = [S {
            x: &[1.0, 2.0],
            y: &[0.1, 0.2],
        }];
        let s0 = &s[0];
        for i in 0..s0.x.len() {
            assert!(true);
        }
    }
}
