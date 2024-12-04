is-photo
========

[![crates.io](https://img.shields.io/crates/v/is-photo.svg)](https://crates.io/crates/is-photo)
[![Documentation](https://docs.rs/is-photo/badge.svg)](https://docs.rs/is-photo)
[![MIT/Apache-2 licensed](https://img.shields.io/crates/l/is-photo.svg)](./LICENSE-Apache)

Utility to determine if an image is likely a photograph or a 2D graphic,
such as a logo, illustration, or digital art.

It does this by taking various statistics from the image and running them
through a pre-trained logistic regression model, along with a few sure-fire heuristics.

On my test set of around 1500 images, it has a 94% accuracy rate. Feel free to submit
links to additional image sets to train on in an issue!

# Example

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let img = image::open("test.jpg")?;

let analysis = is_photo::analyze(&img).expect("Failed to analyze image");

let is_photo = analysis.is_photo(&is_photo::STANDARD_MODEL);
# Ok(()) }
```

Future work may include training on a larger dataset, and possibly using a full neural network
instead of logistic regression.