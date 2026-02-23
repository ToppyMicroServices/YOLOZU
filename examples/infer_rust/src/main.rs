use std::env;
use std::fs;
use std::path::PathBuf;
#[cfg(feature = "onnxruntime")]
use std::process::Command;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Stub,
    OnnxRt,
}

fn parse_mode(raw: &str) -> Option<Mode> {
    match raw {
        "stub" => Some(Mode::Stub),
        "onnxrt" => Some(Mode::OnnxRt),
        _ => None,
    }
}

fn usage(exit_code: i32) -> ! {
    eprintln!(
        "Usage:\n  yolozu_infer_rust --out <path>\n  yolozu_infer_rust --mode onnxrt --onnx <model.onnx> --out <path> [--image <path>] [--input-shape N,C,H,W]\n\nModes:\n  stub   : default, no-deps schema stub writer\n  onnxrt : optional ONNXRuntime smoke runner (requires --features onnxruntime)\n"
    );
    std::process::exit(exit_code);
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    usage(2);
}

#[cfg(feature = "onnxruntime")]
fn json_escape(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 8);
    for c in src.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn parse_input_shape(raw: &str) -> Result<[usize; 4], String> {
    let mut parts = raw.split(',');
    let n = parts
        .next()
        .ok_or_else(|| "input-shape must be N,C,H,W".to_string())?
        .parse::<usize>()
        .map_err(|_| "input-shape N must be positive integer".to_string())?;
    let c = parts
        .next()
        .ok_or_else(|| "input-shape must be N,C,H,W".to_string())?
        .parse::<usize>()
        .map_err(|_| "input-shape C must be positive integer".to_string())?;
    let h = parts
        .next()
        .ok_or_else(|| "input-shape must be N,C,H,W".to_string())?
        .parse::<usize>()
        .map_err(|_| "input-shape H must be positive integer".to_string())?;
    let w = parts
        .next()
        .ok_or_else(|| "input-shape must be N,C,H,W".to_string())?
        .parse::<usize>()
        .map_err(|_| "input-shape W must be positive integer".to_string())?;
    if parts.next().is_some() {
        return Err("input-shape must be N,C,H,W".to_string());
    }
    Ok([n, c, h, w])
}

fn build_stub_payload() -> String {
    "{\n  \"predictions\": []\n}\n".to_string()
}

#[cfg(feature = "onnxruntime")]
fn build_onnxrt_payload(
    images: &[String],
    onnx_path: &str,
    input_shape: [usize; 4],
    output_shapes_json: &str,
) -> String {
    let mut payload = String::from("{\n  \"predictions\": [");
    for (idx, image) in images.iter().enumerate() {
        if idx > 0 {
            payload.push(',');
        }
        payload.push_str("\n    {\n      \"image\": \"");
        payload.push_str(&json_escape(image));
        payload.push_str("\",\n      \"detections\": []\n    }");
    }
    if !images.is_empty() {
        payload.push('\n');
        payload.push_str("  ");
    }
    payload.push_str("],\n  \"meta\": {\n");
    payload.push_str("    \"backend\": \"onnxruntime-rust\",\n");
    payload.push_str("    \"model\": \"");
    payload.push_str(&json_escape(onnx_path));
    payload.push_str("\",\n");
    payload.push_str("    \"extra\": {\n");
    payload.push_str("      \"input_shape\": [");
    payload.push_str(&format!(
        "{},{},{},{}",
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ));
    payload.push_str("],\n");
    payload.push_str("      \"output_shapes\": ");
    payload.push_str(output_shapes_json);
    payload.push('\n');
    payload.push_str("    }\n");
    payload.push_str("  }\n");
    payload.push_str("}\n");
    payload
}

#[cfg(feature = "onnxruntime")]
fn run_onnxruntime(
    onnx_path: &PathBuf,
    input_shape: [usize; 4],
    images: &[String],
) -> Result<String, String> {
    let input_shape_str = format!(
        "{},{},{},{}",
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    );
    let py = r#"
import json
import sys
import numpy as np
import onnxruntime as ort

onnx_path = sys.argv[1]
shape = [int(v) for v in sys.argv[2].split(',')]
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
inputs = {input_name: np.zeros(shape, dtype=np.float32)}
outs = sess.run(None, inputs)
shapes = [list(getattr(o, 'shape', [])) for o in outs]
print(json.dumps(shapes))
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .arg(onnx_path)
        .arg(input_shape_str)
        .output()
        .map_err(|e| format!("failed to launch python3 for onnxruntime runner: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "onnxruntime runner failed (python): {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let output_shapes_json = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let output_shapes_json = if output_shapes_json.starts_with('[') && output_shapes_json.ends_with(']') {
        output_shapes_json
    } else {
        "[]".to_string()
    };

    Ok(build_onnxrt_payload(
        images,
        &onnx_path.to_string_lossy(),
        input_shape,
        &output_shapes_json,
    ))
}

#[cfg(not(feature = "onnxruntime"))]
fn run_onnxruntime(
    _onnx_path: &PathBuf,
    _input_shape: [usize; 4],
    _images: &[String],
) -> Result<String, String> {
    Err(
        "onnxruntime mode requested, but this binary was built without the 'onnxruntime' feature. Rebuild with: cargo build --release --features onnxruntime"
            .to_string(),
    )
}

fn main() {
    let mut args = env::args().skip(1);
    let mut out_path: Option<PathBuf> = None;
    let mut mode = Mode::Stub;
    let mut onnx_path: Option<PathBuf> = None;
    let mut images: Vec<String> = Vec::new();
    let mut input_shape = [1usize, 3usize, 64usize, 64usize];

    while let Some(tok) = args.next() {
        match tok.as_str() {
            "--out" => {
                out_path = args.next().map(PathBuf::from);
            }
            "--mode" => {
                let raw = args.next().unwrap_or_default();
                mode = parse_mode(&raw).unwrap_or_else(|| die("mode must be one of: stub, onnxrt"));
            }
            "--onnx" => {
                onnx_path = args.next().map(PathBuf::from);
            }
            "--image" => {
                let image = args.next().unwrap_or_default();
                if image.is_empty() {
                    die("--image requires a value");
                }
                images.push(image);
            }
            "--input-shape" => {
                let raw = args.next().unwrap_or_default();
                input_shape = parse_input_shape(&raw).unwrap_or_else(|msg| die(&msg));
            }
            "-h" | "--help" => usage(2),
            _ => {
                eprintln!("Unknown arg: {tok}");
                usage(2);
            }
        }
    }

    let Some(out_path) = out_path else { usage(2) };
    let payload = match mode {
        Mode::Stub => build_stub_payload(),
        Mode::OnnxRt => {
            let Some(path) = onnx_path.as_ref() else {
                die("onnxrt mode requires --onnx <model.onnx>");
            };
            match run_onnxruntime(path, input_shape, &images) {
                Ok(payload) => payload,
                Err(msg) => die(&msg),
            }
        }
    };

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    fs::write(&out_path, payload).expect("failed to write predictions.json");
    println!("{}", out_path.to_string_lossy());
}
