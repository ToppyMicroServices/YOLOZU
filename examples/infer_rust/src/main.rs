use std::env;
use std::fs;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("Usage: yolozu_infer_rust --out <path>");
    std::process::exit(2);
}

fn main() {
    let mut args = env::args().skip(1);
    let mut out_path: Option<PathBuf> = None;

    while let Some(tok) = args.next() {
        if tok == "--out" {
            out_path = args.next().map(PathBuf::from);
            continue;
        }
        if tok == "-h" || tok == "--help" {
            usage();
        }
        eprintln!("Unknown arg: {tok}");
        usage();
    }

    let Some(out_path) = out_path else { usage() };

    let payload = "{\n  \"predictions\": []\n}\n".to_string();

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    fs::write(&out_path, payload).expect("failed to write predictions.json");
    println!("{}", out_path.to_string_lossy());
}
