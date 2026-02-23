# gpu-zisn DoD summary

- stage: `all`
- dod_status: `pass`
- zisn1_result: `success`
- zisn2_result: `success`
- zisn3_result: `success`

## checks
- zisn1_model_onnx: ok
- zisn1_model_meta: ok
- zisn1_ort_cuda_check: ok
- zisn2_engine_meta: ok
- zisn2_pred_trt: ok
- zisn2_parity_report: ok
- zisn2_latency: ok
- zisn2_eval: ok
- zisn3_opencv_cpu_pred: ok
- zisn3_opencv_cpu_meta: ok
- zisn3_opencv_cuda_status: ok
- zisn3_ttt_summary: ok
- zisn3_doctor: ok
- zisn3_ttt_tent_log: ok
- zisn3_ttt_cotta_log: ok
- zisn3_ttt_eata_log: ok
- zisn3_ttt_sar_log: ok
- zisn3_opencv_cuda_pred: missing
- zisn3_opencv_cuda_meta: missing
- zisn3_opencv_parity: missing
