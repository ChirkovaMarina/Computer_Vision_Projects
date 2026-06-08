# Module 4b — Learned Features and Matching

This module is a reference area for learned local-feature pipelines such as SuperPoint and SuperGlue.

## Contents

- `README.md` — notes and example commands for external SuperPoint / SuperGlue repositories.

## External References

- SuperPoint: `https://github.com/sadekovlar/super_point_relative_navi`
- SuperGlue: `https://github.com/magicleap/SuperGluePretrainedNetwork`

## Example Commands

```bash
python demo_superpoint.py ./video/get.130.151.left.avi
python matching.py ./video/get.130.151.left.avi
python relative_pose.py

python demo_superglue.py
python demo_superglue.py --input assets/phototourism_sample_images/ --output_dir dump_demo_sequence --resize 320 240
python demo_superglue.py --input assets/freiburg_sequence/ --output_dir dump_demo_sequence --resize 320 240
python matching.py
```
