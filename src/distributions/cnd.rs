/// Cumulative double precision algorithm based on Hart 1968
/// Based on implementation by Graeme West
#[allow(non_snake_case)]
pub fn CND(x: f64) -> f64 {
    let y = x.abs();
    if y > 37.0 {
        return 0.0;
    }

    let e = (-(y * y) / 2.0).exp();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_should_calc_cdf() {
        for (x, expected, threshold) in [
            (0.0, 0.5, 1e-12),
            (0.1, 0.539827837277029, 1e-12),
            (0.2, 0.579259709439103, 1e-12),
            (0.3, 0.6179114221889526, 1e-12),
            (0.4, 0.6554217416103242, 1e-12),
            (0.5, 0.6914624612740131, 1e-12),
            (0.6, 0.7257468822499265, 1e-12),
            (0.7, 0.758036347776927, 1e-12),
            (0.8, 0.7881446014166034, 1e-12),
            (0.9, 0.8159398746532405, 1e-12),
            (1.0, 0.8413447460685429, 1e-12),
        ] {
            let actual = CND(x);
            assert!(
                (expected - actual).abs() <= threshold,
                "cdf({}) -> {} (expected: {}, diff: {:e}, threshold: {:e})",
                x,
                actual,
                expected,
                (expected - actual).abs(),
                threshold
            )
        }
    }
}
