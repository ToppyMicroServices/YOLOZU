use std::env;
use std::fs;
use std::path::PathBuf;

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

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

    let out_str = out_path.to_string_lossy();
    let meta_note = "stub template: replace with real Rust inference backend";
    let payload = format!(
        "{{\n  \"predictions\": [],\n  \"meta\": {{\n    \"backend\": \"rust_stub\",\n    \"note\": \"{}\",\n    \"out\": \"{}\"\n  }}\n}}\n",
        json_escape(meta_note),
        json_escape(&out_str),
    );

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    fs::write(&out_path, payload).expect("failed to write predictions.json");
    println!("{}", out_str);
}

