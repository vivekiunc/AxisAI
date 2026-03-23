# AxisAI
# Spinal Curvature Screener — Postural Cobb Angle Estimator

Estimates spinal curvature from a standard back photo using **MediaPipe Pose** landmark detection and a polynomial curve fitting approach analogous to the clinical Cobb angle method.

---

## ⚠️ Important Disclaimer

This tool estimates **postural curvature**, not a clinical Cobb angle. The traditional Cobb angle requires standing X-ray imaging to visualize vertebral endplates. This tool is a **screening aid only** — it should never replace a physician's evaluation.

---

## How It Works

1. **Pose Detection** — MediaPipe Pose detects body landmarks (ears, shoulders, hips, knees) from the back photo.
2. **Spine Midline** — Midpoints between symmetric pairs are computed to approximate the spine's centerline.
3. **Curve Fitting** — A 2nd-degree polynomial is fit through the midline points.
4. **Cobb Angle** — The angle between the tangent vectors at the top and bottom of the fitted curve is computed (analogous to the Cobb method).
5. **Supplementary Metrics** — Shoulder tilt and hip tilt angles provide additional asymmetry information.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Command-Line Interface

```bash
# Basic usage (saves annotated image automatically)
python cobb_angle_estimator.py path/to/back_photo.jpg

# Save to a specific path and display window
python cobb_angle_estimator.py photo.jpg --output result.jpg --show

# Lower detection confidence for trickier photos
python cobb_angle_estimator.py photo.jpg --confidence 0.3
```

### Web App (Streamlit)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Tips for Best Results

- 📸 Photo taken **directly from behind** the subject
- 🕴 Full torso visible — both shoulders and hips in frame
- 💡 Good, even lighting
- 👕 Minimal or form-fitting clothing
- 📐 Camera at approximately mid-back height, not angled up or down

---

## Severity Classification

| Cobb Angle | Classification |
|---|---|
| < 10° | Normal |
| 10° – 25° | Mild Scoliosis |
| 25° – 40° | Moderate Scoliosis |
| > 40° | Severe Scoliosis |

*(Based on standard clinical scoliosis classification thresholds.)*

---

## Output

- **Annotated image** with spine midline overlay, shoulder/hip reference lines, and HUD panel
- **Console report** (CLI) or metrics panel (web app) showing:
  - Cobb angle estimate
  - Severity classification
  - Shoulder tilt angle
  - Hip tilt angle
  - Detection confidence

---

## Limitations

- Accuracy depends heavily on photo quality and positioning
- Surface landmarks ≠ vertebral endplates — systematic underestimation of true Cobb angle is expected
- Not validated against clinical X-ray measurements in a controlled study
- Should not be used for treatment decisions