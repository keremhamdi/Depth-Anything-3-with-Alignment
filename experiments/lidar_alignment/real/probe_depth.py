"""
Interactive depth probe: click anywhere on an RGB image, get the predicted
depth at that pixel. Useful for spot-checking real-capture predictions
against known/measurable distances.

Usage:
    python probe_depth.py --rgb path/to/rgb.png --depth path/to/depth.npy

Left click: query a point (marks it with a colored dot + depth label)
Right click: remove nearest marker
'r' key: reset all markers
's' key: save annotated image + probe log
'q' key: quit
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class DepthProbe:
    def __init__(self, rgb_path, depth_path, out_dir):
        self.rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        self.depth = np.load(depth_path)
        if self.depth.ndim > 2:
            self.depth = np.squeeze(self.depth)
        assert self.depth.shape == self.rgb.shape[:2], \
            f"Shape mismatch: RGB {self.rgb.shape[:2]}, depth {self.depth.shape}"
        self.H, self.W = self.rgb.shape[:2]
        self.name = Path(rgb_path).stem
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        finite = self.depth[np.isfinite(self.depth) & (self.depth > 0)]
        print(f"Depth range: {finite.min():.3f} - {finite.max():.3f} m")
        print(f"Median depth: {np.median(finite):.3f} m")

        self.probes = []
        self.display = cv2.cvtColor(self.rgb.copy(), cv2.COLOR_RGB2BGR)

    def query(self, x, y, patch=3):
        """Get median depth in a small patch (more robust than one pixel)."""
        y0, y1 = max(0, y-patch), min(self.H, y+patch+1)
        x0, x1 = max(0, x-patch), min(self.W, x+patch+1)
        patch_depths = self.depth[y0:y1, x0:x1]
        finite = patch_depths[np.isfinite(patch_depths) & (patch_depths > 0)]
        if len(finite) == 0:
            return None, None
        median = float(np.median(finite))
        std = float(np.std(finite))
        return median, std

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            median, std = self.query(x, y)
            if median is None:
                print(f"({x},{y}): NO VALID DEPTH")
                return
            probe_id = len(self.probes) + 1
            self.probes.append({
                "id": probe_id, "x": x, "y": y,
                "depth_m": median, "std_m": std,
            })
            print(f"[{probe_id:2d}] pixel ({x:4d},{y:4d}) -> {median:.3f} m "
                  f"(±{std*100:.1f} cm)")
            self.redraw()

        elif event == cv2.EVENT_RBUTTONDOWN and self.probes:
            dists = [(p["x"]-x)**2 + (p["y"]-y)**2 for p in self.probes]
            idx = int(np.argmin(dists))
            removed = self.probes.pop(idx)
            print(f"Removed probe {removed['id']}")
            for i, p in enumerate(self.probes, start=1):
                p["id"] = i
            self.redraw()

    def redraw(self):
        self.display = cv2.cvtColor(self.rgb.copy(), cv2.COLOR_RGB2BGR)
        for p in self.probes:
            color = self.depth_color(p["depth_m"])
            cv2.circle(self.display, (p["x"], p["y"]), 8, (0, 0, 0), -1)
            cv2.circle(self.display, (p["x"], p["y"]), 6, color, -1)
            label = f"{p['id']}:{p['depth_m']:.2f}m"
            self.draw_label(self.display, (p["x"]+10, p["y"]-10), label)

    @staticmethod
    def depth_color(d):
        # Blue = close, red = far, on a 0-5m scale.
        norm = np.clip(d / 5.0, 0, 1)
        b = int(255 * (1 - norm))
        r = int(255 * norm)
        return (b, 128, r)

    @staticmethod
    def draw_label(img, pos, text):
        # Black background + white text for readability.
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = pos
        cv2.rectangle(img, (x-2, y-th-4), (x+tw+2, y+2), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def save(self):
        img_path = self.out_dir / f"{self.name}__probed.png"
        cv2.imwrite(str(img_path), self.display)
        csv_path = self.out_dir / f"{self.name}__probes.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "x", "y", "depth_m", "std_m"])
            w.writeheader()
            w.writerows(self.probes)
        print(f"Saved: {img_path}")
        print(f"Saved: {csv_path}")

    def run(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.name, self.mouse_callback)
        print("\nControls:")
        print("  Left click:  query point (adds probe)")
        print("  Right click: remove nearest probe")
        print("  'r':         reset all probes")
        print("  's':         save annotated image + CSV")
        print("  'q' or Esc:  quit\n")

        while True:
            cv2.imshow(self.name, self.display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self.probes.clear()
                self.redraw()
                print("Reset all probes")
            elif key == ord('s'):
                self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, type=Path)
    ap.add_argument("--depth", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/depth_probes"))
    args = ap.parse_args()

    probe = DepthProbe(args.rgb, args.depth, args.out_dir)
    probe.run()


if __name__ == "__main__":
    main()
