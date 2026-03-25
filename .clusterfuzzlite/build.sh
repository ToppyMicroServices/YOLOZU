#!/usr/bin/env bash
set -euo pipefail

cd /src
python3 -m pip wheel --no-deps --wheel-dir /tmp/clusterfuzzlite-wheels /src
python3 -m pip install --no-deps /tmp/clusterfuzzlite-wheels/*.whl

for fuzzer in $(find fuzz -name '*_fuzzer.py' | sort); do
  fuzzer_name="$(basename -s .py "${fuzzer}")"
  pkg_name="${fuzzer_name}.pkg"
  pyinstaller --distpath "$OUT" --onefile --name "${pkg_name}" "${fuzzer}"
  cat > "$OUT/${fuzzer_name}" <<EOF
#!/bin/sh
# LLVMFuzzerTestOneInput for fuzzer detection.
this_dir=\$(dirname "\$0")
\$this_dir/${pkg_name} "\$@"
EOF
  chmod +x "$OUT/${fuzzer_name}"
done
