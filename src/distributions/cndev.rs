/// Inverse cumulative normal distribution function
#[allow(non_snake_case, non_upper_case_globals, dead_code)]
pub fn cndev(U: f64) -> f64 {
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    const b: [f64; 4] = [
        -8.4735109309,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    const c: [f64; 9] = [
        0.337475482272615,
        0.976169019091719,
        0.160797971491821,
        2.76438810333863E-02,
        3.8405729373609E-03,
        3.951896511919E-04,
        3.21767881767818E-05,
        2.888167364E-07,
        3.960315187E-07,
    ];

    let x = U - 0.5;
    if x.abs() < 0.92 {
        let r = x * x;
        let r = x * (((A[3] * r + A[2]) * r + A[1]) * r + A[0])
            / ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
        return r;
    }

    let r = if x < 0.0 { U } else { 1.0 - U };
    let r = (-r.ln()).ln();
    let r = c[0]
        + r * (c[1]
            + r * (c[2]
                + r * (c[3] + r + (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))));
    if x < 0.0 { -r } else { r }
}
