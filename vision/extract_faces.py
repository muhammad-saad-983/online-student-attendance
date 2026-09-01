from vision.config import load_enrollment_settings
from vision.datasets import extract_faces_for_all


def main() -> None:
    settings = load_enrollment_settings()
    extract_faces_for_all(settings)


if __name__ == "__main__":
    main()
