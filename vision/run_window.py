from __future__ import annotations

import argparse

from vision.windows import RecognizerConfig, WindowRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an attendance window via webcam.")
    parser.add_argument(
        "--window-name",
        type=str,
        default="manual",
        choices=["early", "mid", "manual"],
        help="Name of the window to run.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the FastAPI server.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to sample (default: N from k_of_n in config.yaml).",
    )

    args = parser.parse_args()

    cfg = RecognizerConfig(
        window_name=args.window_name,
        api_url=args.api_url,
        camera_index=args.camera_index,
        num_frames=args.num_frames,
    )

    recognizer = WindowRecognizer(cfg)
    payload = recognizer.run_window()
    recognizer.post_results(payload)


if __name__ == "__main__":
    main()
