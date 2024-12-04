fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("missing path argument");

    let img = image::open(path)?;

    let analysis = is_photo::analyze(&img).expect("failed to analyze image");

    let raw_is_photo = analysis.raw_is_photo(&is_photo::STANDARD_MODEL);

    if raw_is_photo > 0.5 {
        println!(
            "Image is likely a Photo at {:.2}% confidence",
            raw_is_photo * 100.0
        );
    } else {
        println!(
            "Image is likely a Vector Graphic at {:.2}% confidence",
            (1.0 - raw_is_photo) * 100.0
        );
    }

    println!("{:#?}", analysis);

    Ok(())
}
