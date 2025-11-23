use std::cmp;

pub fn comb(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }

    let mut r = 1.0;
    for i in 0..cmp::min(k, n - k) {
        r /= (i + 1) as f64;
        r *= (n - i) as f64;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_should_calc_comb() {
        let actual = comb(5, 2);
        assert_eq!(actual, 10.0);
    }
}
