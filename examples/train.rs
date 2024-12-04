use std::{
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

use is_photo::{analyze, regression::RegressionModel};

use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum Classification {
    Photo,
    Vector,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // manually copied from the previous output
    let mut reg = RegressionModel {
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
        ..Default::default()
    };

    let mut data = Vec::new();
    let correct = AtomicUsize::new(0);

    for (classification, dir) in [
        (Classification::Photo, "./data/photo"),
        (Classification::Vector, "./data/vector"),
    ] {
        let mut entries = Vec::new();

        for file in std::fs::read_dir(dir)? {
            let file = file?;

            if file.file_type()?.is_dir() {
                continue;
            }

            entries.push(file);
        }

        data.append(
            &mut entries
                .into_par_iter()
                .fold(Vec::new, |mut data, file| {
                    let path = file.path();

                    let Ok(img) = image::open(&path) else {
                        eprintln!("Error reading: {}", path.display());
                        return data;
                    };

                    let analysis = analyze(&img).unwrap();

                    let is_vector = reg.predict(&analysis) > 0.5;
                    let is_vector2 = analysis.is_vector(&is_photo::STANDARD_MODEL);

                    if is_vector != is_vector2 {
                        eprintln!(
                            "{} disagrees with analysis: {} vs {}",
                            path.display(),
                            is_vector,
                            is_vector2
                        );
                    }

                    match (classification, is_vector) {
                        (Classification::Photo, true) => {
                            eprintln!("{} is misclassified as a vector", path.display());
                        }
                        (Classification::Vector, false) => {
                            eprintln!("{} is misclassified as a photo", path.display());
                        }
                        _ => {
                            correct.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    data.push((classification, analysis, path));

                    data
                })
                .reduce(Vec::new, |mut a, mut b| {
                    a.append(&mut b);
                    a
                }),
        );
    }

    let correct = correct.load(Ordering::Relaxed);

    println!(
        "Prediction accuracy: {correct}/{} ({}%)",
        data.len(),
        (correct as f64 / data.len() as f64) * 100.0
    );

    let mut out = std::io::BufWriter::new(std::fs::File::create("analysis.csv")?);

    writeln!(out, "Classification,Filename,RGCov,GBCov,BRCov,Mean,MeanSqr,Variance,Entropy,Low,High,Ratio,Skewness,Kurtosis,PeakAmp,PeakFreq,PeakRatio")?;

    for (classification, a, p) in &data {
        let class = match classification {
            Classification::Photo => "Photo",
            Classification::Vector => "Vector",
        };

        write!(
            out,
            "\n{class},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            p.file_name().unwrap().to_string_lossy(),
            a.rg_cov,
            a.gb_cov,
            a.br_cov,
            a.mean,
            a.mean_square,
            a.variance,
            a.entropy,
            a.low_energy,
            a.high_energy,
            a.spectral_energy_ratio,
            a.spectral_skewness,
            a.spectral_kurtosis,
            a.peak_amplitude,
            a.peak_frequency,
            a.peak_to_avg_ratio,
        )?;
    }

    println!("Training regression model");

    drop(out);

    let mut data = data
        .into_iter()
        .map(|(c, a, _)| (a, c == Classification::Vector))
        .collect::<Vec<_>>();

    // shuffle and add noise to things to avoid any potential overfitting

    use rand::seq::SliceRandom;

    data.shuffle(&mut rand::thread_rng());

    reg.bias += rand::random::<f32>() - 0.5;
    for w in reg.weights.iter_mut() {
        *w += (rand::random::<f32>() - 0.5) * 0.5;
    }

    reg.train(&data, 10_000_000);

    println!("{:?} + {}", reg.weights, reg.bias);

    Ok(())
}
