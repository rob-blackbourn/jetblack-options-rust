use libm::{exp, fabs, log};

/// Cumulative double precision algorithm based on Hart 1968
/// Based on implementation by Graeme West
#[allow(non_snake_case)]
pub fn CND(x: f64) -> f64 {
    let y = fabs(x);
    if y > 37.0 {
        return 0.0;
    }

    let e = exp(-(y * y) / 2.0);
    let c = if y < 7.07106781186547 {
        let a = 3.52624965998911E-02 * y + 0.700383064443688;
        let a = a * y + 6.37396220353165;
        let a = a * y + 33.912866078383;
        let a = a * y + 112.079291497871;
        let a = a * y + 221.213596169931;
        let a = a * y + 220.206867912376;
        let b = 8.83883476483184E-02 * y + 1.75566716318264;
        let b = b * y + 16.064177579207;
        let b = b * y + 86.7807322029461;
        let b = b * y + 296.564248779674;
        let b = b * y + 637.333633378831;
        let b = b * y + 793.826512519948;
        let b = b * y + 440.413735824752;
        e * a / b
    } else {
        let a = y + 0.65;
        let a = y + 4.0 / a;
        let a = y + 3.0 / a;
        let a = y + 2.0 / a;
        let a = y + 1.0 / a;
        e / (a * 2.506628274631)
    };

    if x > 0.0 { 1.0 - c } else { c }
}
