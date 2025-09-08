// build.rs
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        winres::WindowsResource::new()
            //.set_icon("your_icon.ico") // Optional: if you have an icon
            .set("InternalName", "NTOOMER")
            .set("OriginalFilename", "ntoomer.exe")
            .compile().unwrap();
    }
}