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
        "Usage:\n  yolozu_infer_rust --out <path>\n  yolozu_infer_rust --mode onnxrt --onnx <model.onnx> --out <path> [--image <path>] [--input-shape N,C,H,W] [--combined-format xyxy_score_class] [--boxes-scale norm|abs] [--input-size WxH] [--min-score 0.001] [--topk 300]\n\nModes:\n  stub   : default, no-deps schema stub writer\n  onnxrt : optional ONNXRuntime runner with declared xyxy_score_class decode (requires --features onnxruntime)\n"
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
fn run_onnxruntime(
    onnx_path: &PathBuf,
    input_shape: [usize; 4],
    images: &[String],
    combined_format: &str,
    boxes_scale: &str,
    input_size: &str,
    min_score: f32,
    topk: usize,
) -> Result<String, String> {
    if combined_format != "xyxy_score_class" {
        return Err(
            "onnxrt mode currently supports --combined-format xyxy_score_class only".to_string(),
        );
    }
    if boxes_scale != "norm" && boxes_scale != "abs" {
        return Err("--boxes-scale must be one of: norm, abs".to_string());
    }
    let input_shape_str = format!(
        "{},{},{},{}",
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    );
    let images_json = format!(
        "[{}]",
        images
            .iter()
            .map(|image| format!("\"{}\"", json_escape(image)))
            .collect::<Vec<String>>()
            .join(",")
    );
    let py = r#"
import json
import sys
import numpy as np
import onnxruntime as ort

onnx_path = sys.argv[1]
shape = [int(v) for v in sys.argv[2].split(',')]
images = json.loads(sys.argv[3])
combined_format = sys.argv[4]
boxes_scale = sys.argv[5]
input_size = sys.argv[6]
min_score = float(sys.argv[7])
topk = int(sys.argv[8])

if combined_format != 'xyxy_score_class':
    raise SystemExit('unsupported combined format')
try:
    input_w, input_h = [int(v) for v in input_size.lower().split('x', 1)]
except Exception as exc:
    raise SystemExit(f'input-size must be WxH: {input_size}') from exc
if input_w <= 0 or input_h <= 0:
    raise SystemExit('input-size dims must be positive')

sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
inputs = {input_name: np.zeros(shape, dtype=np.float32)}
outs = sess.run(None, inputs)
shapes = [list(getattr(o, 'shape', [])) for o in outs]

detections = []
if outs:
    arr = np.asarray(outs[0], dtype=np.float32)
    if arr.size:
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] < 6:
            raise SystemExit(f'xyxy_score_class decode expects last dimension >= 6, got {arr.shape}')
        for row in arr:
            x1, y1, x2, y2, score, class_id = [float(v) for v in row[:6]]
            if score < min_score:
                continue
            if boxes_scale == 'abs':
                cx = ((x1 + x2) / 2.0) / float(input_w)
                cy = ((y1 + y2) / 2.0) / float(input_h)
                bw = max(0.0, x2 - x1) / float(input_w)
                bh = max(0.0, y2 - y1) / float(input_h)
            else:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
            detections.append({
                'class_id': int(class_id),
                'score': float(score),
                'bbox': {'cx': float(cx), 'cy': float(cy), 'w': float(bw), 'h': float(bh)},
            })
detections.sort(key=lambda item: float(item['score']), reverse=True)
if topk > 0:
    detections = detections[:topk]

payload = {
    'predictions': [
        {'image': image, 'detections': list(detections)}
        for image in images
    ],
    'meta': {
        'backend': 'onnxruntime-rust',
        'model': onnx_path,
        'extra': {
            'input_shape': shape,
            'input_size': [input_w, input_h],
            'combined_format': combined_format,
            'boxes_scale': boxes_scale,
            'output_shapes': shapes,
            'decoded_detections_per_image': len(detections),
        },
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .arg(onnx_path)
        .arg(input_shape_str)
        .arg(images_json)
        .arg(combined_format)
        .arg(boxes_scale)
        .arg(input_size)
        .arg(min_score.to_string())
        .arg(topk.to_string())
        .output()
        .map_err(|e| format!("failed to launch python3 for onnxruntime runner: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "onnxruntime runner failed (python): {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string() + "\n")
}

#[cfg(not(feature = "onnxruntime"))]
fn run_onnxruntime(
    _onnx_path: &PathBuf,
    _input_shape: [usize; 4],
    _images: &[String],
    _combined_format: &str,
    _boxes_scale: &str,
    _input_size: &str,
    _min_score: f32,
    _topk: usize,
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
    let mut combined_format = "xyxy_score_class".to_string();
    let mut boxes_scale = "norm".to_string();
    let mut input_size = "64x64".to_string();
    let mut min_score = 0.001f32;
    let mut topk = 300usize;

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
            "--combined-format" => {
                combined_format = args.next().unwrap_or_default();
                if combined_format.is_empty() {
                    die("--combined-format requires a value");
                }
            }
            "--boxes-scale" => {
                boxes_scale = args.next().unwrap_or_default();
                if boxes_scale != "norm" && boxes_scale != "abs" {
                    die("--boxes-scale must be one of: norm, abs");
                }
            }
            "--input-size" => {
                input_size = args.next().unwrap_or_default();
                if input_size.is_empty() {
                    die("--input-size requires a value");
                }
            }
            "--min-score" => {
                let raw = args.next().unwrap_or_default();
                min_score = raw
                    .parse::<f32>()
                    .unwrap_or_else(|_| die("--min-score must be a number"));
            }
            "--topk" => {
                let raw = args.next().unwrap_or_default();
                topk = raw
                    .parse::<usize>()
                    .unwrap_or_else(|_| die("--topk must be a non-negative integer"));
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
            match run_onnxruntime(
                path,
                input_shape,
                &images,
                &combined_format,
                &boxes_scale,
                &input_size,
                min_score,
                topk,
            ) {
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
