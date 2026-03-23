"""
Postural Cobb Angle Estimator from Back Photos
================================================
Uses MediaPipe Pose to detect body landmarks from a back-facing photo,
then estimates spinal curvature approximation using a postural Cobb angle
method based on shoulder, hip, and spine midline landmarks.

DISCLAIMER: This is a SCREENING TOOL only. It estimates postural curvature,
NOT a clinical Cobb angle (which requires X-ray). Do not use for diagnosis.
"""

import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Optional
import argparse
import os


# ─────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────

@dataclass
class LandmarkPoint:
    x: float
    y: float
    visibility: float

    def to_pixel(self, w: int, h: int):
        return int(self.x * w), int(self.y * h)


@dataclass
class SpineAnalysisResult:
    cobb_angle: float
    severity: str
    severity_color: tuple
    shoulder_tilt_deg: float
    hip_tilt_deg: float
    spine_points: list
    landmarks_raw: dict
    annotated_image: np.ndarray
    confidence: float


# ─────────────────────────────────────────
# Cobb Angle Estimator Class
# ─────────────────────────────────────────

class CobbAngleEstimator:
    """
    Estimates postural spinal curvature from a back photo using pose landmarks.

    Method:
    -------
    1. Detect body landmarks with MediaPipe Pose
    2. Extract shoulder midpoint, hip midpoint, and lateral landmarks
    3. Fit a spine midline curve through available midpoints
    4. Compute the angle between the tangents at the top and bottom of the curve
       (analogous to the Cobb angle method)
    5. Incorporate shoulder and hip tilt as supplementary metrics
    """

    SEVERITY_THRESHOLDS = {
        "Normal": (0, 10),
        "Mild Scoliosis": (10, 25),
        "Moderate Scoliosis": (25, 40),
        "Severe Scoliosis": (40, float("inf")),
    }

    SEVERITY_COLORS = {
        "Normal": (60, 200, 100),           # Green
        "Mild Scoliosis": (50, 200, 255),   # Yellow
        "Moderate Scoliosis": (30, 140, 255),  # Orange
        "Severe Scoliosis": (50, 50, 230),  # Red
    }

    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        )

    def _extract_landmarks(self, results, img_w: int, img_h: int) -> Optional[dict]:
        """Extract and validate required landmarks."""
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        required = [
            PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER,
            PL.LEFT_HIP, PL.RIGHT_HIP,
        ]

        extracted = {}
        for point in required:
            l = lm[point.value]
            extracted[point.name] = LandmarkPoint(l.x, l.y, l.visibility)
            if l.visibility < 0.3:
                return None  # Not confident enough

        # Optional midpoints
        for point in [PL.LEFT_EAR, PL.RIGHT_EAR, PL.LEFT_KNEE, PL.RIGHT_KNEE]:
            l = lm[point.value]
            extracted[point.name] = LandmarkPoint(l.x, l.y, l.visibility)

        return extracted

    def _compute_midpoint(self, p1: LandmarkPoint, p2: LandmarkPoint) -> tuple:
        return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    def _angle_between_lines(self, p1, p2, p3, p4) -> float:
        """
        Compute angle between line p1->p2 and line p3->p4 in degrees.
        Returns the acute angle.
        """
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])

        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = math.degrees(math.acos(abs(cos_theta)))
        return angle

    def _horizontal_tilt(self, left: LandmarkPoint, right: LandmarkPoint) -> float:
        """Compute tilt angle from horizontal for a pair of symmetric landmarks."""
        dx = right.x - left.x
        dy = right.y - left.y
        return math.degrees(math.atan2(dy, dx))

    def _build_spine_points(self, landmarks: dict, img_w: int, img_h: int) -> list:
        """
        Build a list of spine midline points from top to bottom.
        Uses ear midpoint (head), shoulder midpoint, hip midpoint, knee midpoint.
        """
        pts = []
        PL = self.mp_pose.PoseLandmark

        def mid_px(name1, name2):
            l1 = landmarks.get(name1)
            l2 = landmarks.get(name2)
            if l1 and l2:
                mx = (l1.x + l2.x) / 2 * img_w
                my = (l1.y + l2.y) / 2 * img_h
                return (int(mx), int(my))
            return None

        # From top to bottom
        ear_mid = mid_px("LEFT_EAR", "RIGHT_EAR")
        shoulder_mid = mid_px("LEFT_SHOULDER", "RIGHT_SHOULDER")
        hip_mid = mid_px("LEFT_HIP", "RIGHT_HIP")
        knee_mid = mid_px("LEFT_KNEE", "RIGHT_KNEE")

        for pt in [ear_mid, shoulder_mid, hip_mid, knee_mid]:
            if pt:
                pts.append(pt)

        return pts

    def _fit_cobb_angle(self, spine_points: list) -> float:
        """
        Estimate the Cobb angle from a sequence of spine midpoints.

        Uses the angle between:
        - The line from top two points (upper segment)
        - The line from bottom two points (lower segment)

        If more than 4 points available, uses polynomial fit tangents.
        """
        if len(spine_points) < 2:
            return 0.0

        pts = np.array(spine_points, dtype=float)

        if len(pts) == 2:
            # Only two points: can only compute tilt from vertical
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            return abs(math.degrees(math.atan2(dx, dy)))

        if len(pts) == 3:
            # Use angle at middle point
            v1 = pts[0] - pts[1]
            v2 = pts[2] - pts[1]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle = 180 - math.degrees(math.acos(np.clip(cos_a, -1, 1)))
            return abs(angle)

        # >= 4 points: use polynomial fit and compute tangent angle at ends
        x = pts[:, 0]
        y = pts[:, 1]

        # Fit x as function of y (spine runs top-to-bottom, y increases downward)
        # Use degree 2 polynomial
        coeffs = np.polyfit(y, x, 2)
        poly = np.poly1d(coeffs)
        dpoly = poly.deriv()

        y_top = y[0]
        y_bot = y[-1]

        # Tangent vectors at top and bottom
        dx_top = dpoly(y_top)
        dx_bot = dpoly(y_bot)

        # Each tangent: direction (dx, 1) normalized (since dy=1 by parameterization)
        v_top = np.array([dx_top, 1.0])
        v_bot = np.array([dx_bot, 1.0])

        cos_a = np.dot(v_top, v_bot) / (np.linalg.norm(v_top) * np.linalg.norm(v_bot) + 1e-9)
        angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
        return abs(angle)

    def _classify_severity(self, angle: float) -> tuple:
        for name, (lo, hi) in self.SEVERITY_THRESHOLDS.items():
            if lo <= angle < hi:
                return name, self.SEVERITY_COLORS[name]
        return "Severe Scoliosis", self.SEVERITY_COLORS["Severe Scoliosis"]

    def _compute_confidence(self, landmarks: dict) -> float:
        """Average visibility of key landmarks as a confidence proxy."""
        keys = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        vis = [landmarks[k].visibility for k in keys if k in landmarks]
        return float(np.mean(vis)) if vis else 0.0

    def _draw_annotations(
        self,
        image: np.ndarray,
        spine_points: list,
        result: dict,
        landmarks: dict,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Draw spine overlay, angle arcs, and stats on the image."""
        overlay = image.copy()

        # ── Draw spine curve ──────────────────────────────────
        if len(spine_points) >= 2:
            for i in range(len(spine_points) - 1):
                cv2.line(overlay, spine_points[i], spine_points[i + 1],
                         (255, 220, 0), 3, cv2.LINE_AA)
            for pt in spine_points:
                cv2.circle(overlay, pt, 7, (255, 255, 255), -1)
                cv2.circle(overlay, pt, 7, (255, 200, 0), 2)

        # ── Draw shoulder line ────────────────────────────────
        ls = landmarks.get("LEFT_SHOULDER")
        rs = landmarks.get("RIGHT_SHOULDER")
        if ls and rs:
            lsp = ls.to_pixel(img_w, img_h)
            rsp = rs.to_pixel(img_w, img_h)
            cv2.line(overlay, lsp, rsp, (100, 220, 255), 2, cv2.LINE_AA)

        # ── Draw hip line ─────────────────────────────────────
        lh = landmarks.get("LEFT_HIP")
        rh = landmarks.get("RIGHT_HIP")
        if lh and rh:
            lhp = lh.to_pixel(img_w, img_h)
            rhp = rh.to_pixel(img_w, img_h)
            cv2.line(overlay, lhp, rhp, (180, 100, 255), 2, cv2.LINE_AA)

        # ── Blend overlay ─────────────────────────────────────
        annotated = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)

        # ── Draw HUD panel ────────────────────────────────────
        severity = result["severity"]
        color = result["severity_color"]
        cobb = result["cobb_angle"]
        sh_tilt = result["shoulder_tilt"]
        hip_tilt = result["hip_tilt"]
        conf = result["confidence"]

        panel_x, panel_y = 20, 20
        panel_w, panel_h = 340, 180
        panel = annotated[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w].copy()
        dark = np.zeros_like(panel)
        panel = cv2.addWeighted(panel, 0.3, dark, 0.7, 0)
        annotated[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = panel

        def put(txt, y_off, scale=0.7, col=(255, 255, 255), bold=False):
            thickness = 2 if bold else 1
            cv2.putText(annotated, txt,
                        (panel_x + 12, panel_y + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, thickness, cv2.LINE_AA)

        put("SPINAL CURVATURE ANALYSIS", 28, 0.65, (200, 200, 200))
        cv2.line(annotated, (panel_x+10, panel_y+36), (panel_x+panel_w-10, panel_y+36),
                 (80, 80, 80), 1)
        put(f"Cobb Angle:   {cobb:.1f} deg", 62, 0.75, (255, 255, 255), bold=True)
        put(severity, 90, 0.7, color, bold=True)
        put(f"Shoulder Tilt: {abs(sh_tilt):.1f} deg", 120, 0.6)
        put(f"Hip Tilt:      {abs(hip_tilt):.1f} deg", 145, 0.6)
        put(f"Confidence:   {conf*100:.0f}%", 168, 0.55, (160, 160, 160))

        # ── Severity bar ──────────────────────────────────────
        bar_x = panel_x + panel_w + 10
        bar_y = panel_y
        bar_h = panel_h
        bar_w = 18
        # gradient bar
        for i in range(bar_h):
            t = i / bar_h
            c = (
                int(60 + t * 170),
                int(200 - t * 160),
                int(100 - t * 50),
            )
            cv2.line(annotated, (bar_x, bar_y+i), (bar_x+bar_w, bar_y+i), c, 1)
        # indicator
        clamped = min(cobb / 50.0, 1.0)
        ind_y = int(bar_y + clamped * bar_h)
        cv2.rectangle(annotated, (bar_x-3, ind_y-4), (bar_x+bar_w+3, ind_y+4),
                      (255, 255, 255), 2)

        # ── Disclaimer ───────────────────────────────────────
        disc = "SCREENING ONLY - Not a clinical diagnosis"
        cv2.putText(annotated, disc,
                    (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        return annotated

    def analyze(self, image_path: str) -> Optional[SpineAnalysisResult]:
        """
        Main analysis pipeline.

        Parameters
        ----------
        image_path : str
            Path to the back photo (JPG/PNG).

        Returns
        -------
        SpineAnalysisResult or None if pose could not be detected.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        img_h, img_w = image.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        landmarks = self._extract_landmarks(results, img_w, img_h)
        if landmarks is None:
            print("[!] Could not detect body landmarks with sufficient confidence.")
            print("    Tips: Use a well-lit photo, full-body or torso visible from the back.")
            return None

        # ── Compute metrics ───────────────────────────────────
        spine_points = self._build_spine_points(landmarks, img_w, img_h)
        cobb_angle = self._fit_cobb_angle(spine_points)

        shoulder_tilt = self._horizontal_tilt(
            landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"]
        )
        hip_tilt = self._horizontal_tilt(
            landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"]
        )

        severity, severity_color = self._classify_severity(cobb_angle)
        confidence = self._compute_confidence(landmarks)

        # ── Annotate image ────────────────────────────────────
        annotated = self._draw_annotations(
            image, spine_points,
            {
                "cobb_angle": cobb_angle,
                "severity": severity,
                "severity_color": severity_color,
                "shoulder_tilt": shoulder_tilt,
                "hip_tilt": hip_tilt,
                "confidence": confidence,
            },
            landmarks, img_w, img_h,
        )

        return SpineAnalysisResult(
            cobb_angle=cobb_angle,
            severity=severity,
            severity_color=severity_color,
            shoulder_tilt_deg=shoulder_tilt,
            hip_tilt_deg=hip_tilt,
            spine_points=spine_points,
            landmarks_raw=landmarks,
            annotated_image=annotated,
            confidence=confidence,
        )

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────

def print_report(result: SpineAnalysisResult, image_path: str):
    sep = "─" * 50
    print(f"\n{sep}")
    print("  SPINAL CURVATURE SCREENING REPORT")
    print(f"{sep}")
    print(f"  Image        : {os.path.basename(image_path)}")
    print(f"  Cobb Angle   : {result.cobb_angle:.2f}°")
    print(f"  Classification: {result.severity}")
    print(f"  Shoulder Tilt: {abs(result.shoulder_tilt_deg):.2f}°")
    print(f"  Hip Tilt     : {abs(result.hip_tilt_deg):.2f}°")
    print(f"  Confidence   : {result.confidence * 100:.0f}%")
    print(f"{sep}")
    print("  ⚠  DISCLAIMER: This is a postural estimate only.")
    print("     It does NOT replace clinical X-ray Cobb measurement.")
    print("     Consult a physician for any medical concern.")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate spinal curvature (postural Cobb angle) from a back photo."
    )
    parser.add_argument("image", help="Path to the back photo")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save the annotated image (default: <input>_annotated.jpg)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the annotated image in a window"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="Minimum landmark detection confidence (0.0–1.0, default: 0.5)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] File not found: {args.image}")
        return

    output_path = args.output or os.path.splitext(args.image)[0] + "_annotated.jpg"

    print(f"[*] Analyzing: {args.image}")

    with CobbAngleEstimator(min_detection_confidence=args.confidence) as estimator:
        result = estimator.analyze(args.image)

    if result is None:
        print("[ERROR] Analysis failed. Could not detect pose landmarks.")
        return

    print_report(result, args.image)
    cv2.imwrite(output_path, result.annotated_image)
    print(f"[✓] Annotated image saved to: {output_path}")

    if args.show:
        cv2.imshow("Spinal Curvature Analysis", result.annotated_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()