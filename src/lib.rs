#![doc = include_str!("../README.md")]
#![no_std]

pub mod regression;

use regression::RegressionModel;

/// Pre-trained regression model with about 94% accuracy on my test set
/// of 1500 images.
pub const STANDARD_MODEL: RegressionModel = RegressionModel {
    weights: [
        -0.08685173,
        -0.041326188,
        -0.09524358,
        -0.16007994,
        -0.8713319,
        0.08773025,
        -0.00017104072,
        -0.00025994,
        -1.0389963,
        -0.08185195,
        -0.050398663,
        0.30969876,
        0.11085072,
        -0.008571971,
    ],
    bias: 6.0718517,
    lr: 0.01,
    lambda: 0.01,
};

/// Image analysis results, most of which are used as features for the regression model
/// to classify images as photos or vectors.
#[derive(Debug)]
pub struct Analysis {
    /// Red-Green Covariance
    pub rg_cov: f32,
    /// Green-Blue Covariance
    pub gb_cov: f32,
    /// Blue-Red Covariance
    pub br_cov: f32,
    /// Image histogram entropy
    pub entropy: f32,
    //pub edginess: f32,
    /// Low frequency energy
    pub low_energy: f32,
    /// High frequency energy
    pub high_energy: f32,
    /// Ratio of low to high frequency energy
    pub spectral_energy_ratio: f32,
    /// Mean spectral energy
    pub spectral_mean: f32,
    pub spectral_mean_square: f32,
    /// Image spectral variance
    pub spectral_variance: f32,
    /// Skewness of the spectral energy distribution
    pub spectral_skewness: f32,
    /// Kurtosis of the spectral energy distribution
    pub spectral_kurtosis: f32,
    /// Amplitude of the peak frequency
    pub peak_amplitude: f32,
    /// Frequency of the peak amplitude, biased towards higher frequencies
    pub peak_frequency: u8,
    /// Ratio of peak amplitude to mean amplitude
    pub peak_to_avg_ratio: f32,
}

impl Analysis {
    pub fn std_dev(&self) -> f32 {
        self.spectral_variance.sqrt()
    }

    /// Returns the raw probability that the image is a vector image.
    pub fn raw_is_vector(&self, model: &RegressionModel) -> f32 {
        // almost always a vector image
        if self.peak_frequency > 24 || self.spectral_kurtosis < 0.0 || self.spectral_skewness < 0.0
        {
            return 1.0;
        }

        model.predict(self)
    }

    /// Returns the raw probability that the image is a photo.
    pub fn raw_is_photo(&self, model: &RegressionModel) -> f32 {
        1.0 - self.raw_is_vector(model).min(1.0)
    }

    /// Returns true if the image is likely a photo, false if it's likely a vector.
    pub fn is_photo(&self, model: &RegressionModel) -> bool {
        !self.is_vector(model)
    }

    /// Returns true if the image is likely a vector, false if it's likely a photo.
    pub fn is_vector(&self, model: &RegressionModel) -> bool {
        self.raw_is_vector(model) > 0.5
    }
}

use image::{GenericImageView, Pixel};

// /// Laplacian operator with diagonals
// #[rustfmt::skip]
// const LAPLACIAN: [f32; 9] = [
//     -1.0, -1.0, -1.0,
//     -1.0,  8.0, -1.0,
//     -1.0, -1.0, -1.0,
// ];

// /// Fast 3x3 image convolution, same as image's filter3x3 but for RGB_8 only
// /// and without any intermediate buffers, outputting to a callback function instead
// #[inline]
// fn apply_kernel<I, P, F>(image: &I, mut kernel: [f32; 9], mut cb: F)
// where
//     I: GenericImageView<Pixel = P>,
//     P: Pixel<Subpixel = u8>,
//     F: FnMut(u32, u32, [f32; 3]),
// {
//     #[rustfmt::skip]
//     const TAPS: &[(isize, isize); 9] = &[
//         (-1, -1), (0, -1), (1, -1),
//         (-1,  0), (0,  0), (1,  0),
//         (-1,  1), (0,  1), (1,  1),
//     ];

//     // apply u8 -> f32 weight here
//     for k in &mut kernel {
//         *k /= 255.0;
//     }

//     let (width, height) = image.dimensions();

//     for y in 1..height - 1 {
//         for x in 1..width - 1 {
//             let mut t = [0.0f32; 3];

//             for (&k, &(a, b)) in kernel.iter().zip(TAPS) {
//                 let x0 = x as isize + a;
//                 let y0 = y as isize + b;

//                 let p = image.get_pixel(x0 as u32, y0 as u32);

//                 for (&c, f) in p.channels().iter().zip(&mut t) {
//                     *f += k * c as f32;
//                 }
//             }

//             cb(x, y, t);
//         }
//     }
// }

#[rustfmt::skip]
#[inline]
fn luma([r, g, b]: [f32; 3]) -> f32 {
    0.212671 * r +
    0.715160 * g +
    0.072169 * b
}

/// Analyze an image and return the results.
///
/// Returns `None` if the image is too small to analyze.
pub fn analyze<I, P>(img: &I) -> Option<Analysis>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = u8>,
{
    use easyfft::{const_size::FftMut, num_complex::Complex};

    let (width, height) = img.dimensions();
    let num_pixels = width as u64 * height as u64;

    if num_pixels < 2 {
        return None;
    }

    let inverse_n = 1.0 / num_pixels as f32;

    // let mut edginess = 0.0;

    // apply_kernel(img, LAPLACIAN, |_, _, edge| {
    //     edginess += inverse_n * 2.0 * (luma(edge) - 0.5).max(0.0);
    // });

    let mut histogram = [Complex::<f32>::ZERO; 256];

    let mut red = 0.0;
    let mut green = 0.0;
    let mut blue = 0.0;

    let mut rg_cov = 0.0;
    let mut gb_cov = 0.0;
    let mut br_cov = 0.0;

    let [r0, g0, b0] = img.get_pixel(0, 0).to_rgb().0.map(|c| c as f32 / 255.0);

    for (_, _, pixel) in img.pixels() {
        let [r, g, b] = pixel.to_rgb().0.map(|c| c as f32 / 255.0);

        histogram[(luma([r, g, b]) * 255.0) as usize].re += 1.0;

        let dr = r - r0;
        let dg = g - g0;
        let db = b - b0;

        red += dr;
        green += dg;
        blue += db;

        rg_cov += dr * dg;
        gb_cov += dg * db;
        br_cov += db * dr;
    }

    rg_cov = (rg_cov - red * green * inverse_n) * inverse_n;
    gb_cov = (gb_cov - green * blue * inverse_n) * inverse_n;
    br_cov = (br_cov - blue * green * inverse_n) * inverse_n;

    // loop through the histogram to normalize and compute entropy
    let mut entropy = 0.0;
    for h in &mut histogram {
        h.re *= inverse_n; // normalize

        if h.re > 0.0 {
            entropy -= h.re * h.re.ln();
        }
    }

    // in-place real FFT
    histogram.fft_mut();

    let mut n = 0.0;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    let mut peak = (0, 0.0);

    let mut low_sum = 0.0;
    let mut high_sum = 0.0;

    // iterate over positive frequencies, excluding the DC offset
    for (idx, s) in histogram[1..128].iter().enumerate() {
        let x = s.norm();

        // allow equal to bias result towards higher frequencies
        if x >= peak.1 {
            peak = (idx, x);
        }

        if idx >= 32 {
            high_sum += x;
        } else {
            low_sum += x;
        }

        let n1 = n;
        n += 1.0;

        let delta = x - mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;

        mean += delta_n;
        m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3;
        m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * m2;
        m2 += term1;
    }

    let variance = m2 / (n - 1.0);
    let skewness = n.sqrt() * m3 / (m2 * m2 * m2).sqrt();
    let kurtosis = (n * m4) / (m2 * m2) - 3.0;

    Some(Analysis {
        rg_cov,
        gb_cov,
        br_cov,
        spectral_mean: mean,
        spectral_mean_square: m2,
        spectral_variance: variance,
        entropy,
        //edginess,
        low_energy: low_sum * inverse_n,
        high_energy: high_sum * inverse_n,
        spectral_energy_ratio: low_sum / high_sum,
        spectral_skewness: skewness,
        spectral_kurtosis: kurtosis,
        peak_amplitude: peak.1,
        peak_frequency: peak.0 as u8,
        peak_to_avg_ratio: peak.1 / mean,
    })
}
