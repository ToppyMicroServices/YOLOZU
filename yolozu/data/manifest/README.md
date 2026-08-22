# Packaged manifests

This folder contains JSON manifests shipped inside the `yolozu` wheel.

- `tools_manifest.json`: a copy of the repository tool manifest (`tools/manifest.json`) for automation and discovery.
- `adaptive_vision_roadmap.json`: a dated public projection of the future Experimental adaptive local-vision scope; Beads remains the live task source of truth.

Access via:

```bash
yolozu resources list
yolozu resources cat manifest/tools_manifest.json
```
