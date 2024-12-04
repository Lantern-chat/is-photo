use crate::Analysis;

const NUM_FEATURES: usize = 14;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionModel {
    pub weights: [f32; NUM_FEATURES],
    pub bias: f32,
    pub lr: f32,
    pub lambda: f32, // regularization strength
}

impl Default for RegressionModel {
    fn default() -> Self {
        RegressionModel {
            weights: [0.0; NUM_FEATURES],
            bias: 0.0,
            lr: 0.01,
            lambda: 0.01,
        }
    }
}

fn sigmoid32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl RegressionModel {
    fn analysis_to_features(a: &Analysis) -> [f32; NUM_FEATURES] {
        [
            //a.edginess,
            a.rg_cov,
            a.gb_cov,
            a.br_cov,
            a.mean,
            a.entropy,
            a.variance.sqrt(), // standard deviation
            a.low_energy,
            a.high_energy,
            a.spectral_energy_ratio,
            a.spectral_skewness,
            a.spectral_kurtosis,
            a.peak_amplitude,
            a.peak_frequency as f32 / 255.0,
            a.peak_to_avg_ratio,
        ]
    }

    fn predict_inner(&self, a: &[f32; NUM_FEATURES]) -> f32 {
        let mut z = self.bias;
        for (&w, &a) in self.weights.iter().zip(a) {
            z += w * a;
        }

        sigmoid32(z)
    }

    pub fn predict(&self, a: &Analysis) -> f32 {
        self.predict_inner(&Self::analysis_to_features(a))
    }

    pub fn train(&mut self, data: &[(Analysis, bool)], iterations: usize) {
        let inverse_n = 1.0 / data.len() as f64;
        let mut weights = self.weights.map(|w| w as f64);
        let mut bias = self.bias as f64;

        let lr = self.lr as f64;
        let lambda = self.lambda as f64;

        for _ in 0..iterations {
            let mut weight_gradients = [0.0; NUM_FEATURES];
            let mut bias_gradient = 0.0;

            for (a, is_vector) in data {
                let a = Self::analysis_to_features(a);

                let prediction = {
                    let mut z = bias;
                    for (&w, a) in weights.iter().zip(a) {
                        z += w * a as f64;
                    }

                    sigmoid64(z)
                };
                let error = prediction - *is_vector as i32 as f64;

                for (wg, a) in weight_gradients.iter_mut().zip(a) {
                    *wg += error * a as f64;
                }

                bias_gradient += error;
            }

            let old_weights = weights;
            let old_bias = bias;

            weights
                .iter_mut()
                .zip(weight_gradients)
                .for_each(|(weight, weight_gradient)| {
                    *weight -= lr * (weight_gradient * inverse_n + lambda * *weight);
                });

            bias -= lr * bias_gradient * inverse_n;

            if bias == old_bias && old_weights == weights {
                // println!("Converged after {i} iterations");
                break;
            }
        }

        self.bias = bias as f32;
        self.weights = weights.map(|w| w as f32);
    }
}
