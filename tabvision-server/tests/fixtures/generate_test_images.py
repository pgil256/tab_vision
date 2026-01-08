"""Generate synthetic test images for fretboard and hand detection tests."""
import cv2
import numpy as np
import os


def create_fretboard_image(
    width: int = 640,
    height: int = 480,
    num_frets: int = 5,
    output_path: str | None = None
) -> np.ndarray:
    """Create a synthetic fretboard image with clear lines.

    Args:
        width: Image width
        height: Image height
        num_frets: Number of fret lines to draw
        output_path: Optional path to save image

    Returns:
        BGR numpy array with synthetic fretboard
    """
    # Create brown/wooden background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 60, 80)  # Dark brown BGR

    # Fretboard region
    fb_left = int(width * 0.15)
    fb_right = int(width * 0.85)
    fb_top = int(height * 0.3)
    fb_bottom = int(height * 0.7)

    # Draw fretboard background (lighter brown)
    cv2.rectangle(
        image,
        (fb_left, fb_top),
        (fb_right, fb_bottom),
        (60, 90, 120),
        -1
    )

    # Draw fret lines (horizontal, silver/white)
    fret_spacing = (fb_right - fb_left) / (num_frets + 1)
    for i in range(num_frets + 2):
        x = int(fb_left + i * fret_spacing)
        cv2.line(
            image,
            (x, fb_top),
            (x, fb_bottom),
            (200, 200, 200),  # Silver
            2
        )

    # Draw string lines (horizontal across fretboard, representing strings)
    for i in range(6):
        y = int(fb_top + (i + 0.5) * (fb_bottom - fb_top) / 6)
        cv2.line(
            image,
            (fb_left, y),
            (fb_right, y),
            (180, 180, 180),  # Lighter gray for strings
            1
        )

    # Draw fretboard edge lines (vertical boundaries)
    cv2.line(image, (fb_left, fb_top), (fb_left, fb_bottom), (100, 100, 100), 3)
    cv2.line(image, (fb_right, fb_top), (fb_right, fb_bottom), (100, 100, 100), 3)

    # Draw top and bottom edges
    cv2.line(image, (fb_left, fb_top), (fb_right, fb_top), (100, 100, 100), 3)
    cv2.line(image, (fb_left, fb_bottom), (fb_right, fb_bottom), (100, 100, 100), 3)

    if output_path:
        cv2.imwrite(output_path, image)

    return image


def create_no_fretboard_image(
    width: int = 640,
    height: int = 480,
    output_path: str | None = None
) -> np.ndarray:
    """Create an image without any fretboard features.

    Args:
        width: Image width
        height: Image height
        output_path: Optional path to save image

    Returns:
        BGR numpy array with no fretboard
    """
    # Create a plain room-like background with no straight lines
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add random noise to simulate a blurry background
    noise = np.random.randint(80, 150, (height, width, 3), dtype=np.uint8)
    image = noise

    # Add some gentle gradient
    for y in range(height):
        brightness = int(100 + 50 * (y / height))
        image[y, :] = np.clip(image[y, :].astype(int) + brightness - 125, 0, 255)

    if output_path:
        cv2.imwrite(output_path, image)

    return image


def create_no_hand_image(
    width: int = 640,
    height: int = 480,
    output_path: str | None = None
) -> np.ndarray:
    """Create an image without any hand features.

    Uses the same no_fretboard image since it's just a generic background.

    Args:
        width: Image width
        height: Image height
        output_path: Optional path to save image

    Returns:
        BGR numpy array with no hand
    """
    return create_no_fretboard_image(width, height, output_path)


def create_simple_video(
    output_path: str,
    duration_seconds: float = 1.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    color: tuple[int, int, int] = (100, 100, 100)
) -> str:
    """Create a simple solid-color video for testing.

    Args:
        output_path: Path to save video
        duration_seconds: Video duration
        fps: Frames per second
        width: Video width
        height: Video height
        color: BGR color tuple

    Returns:
        Path to created video
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color

    num_frames = int(duration_seconds * fps)
    for _ in range(num_frames):
        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    # Generate test images when run directly
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))

    fretboard_path = os.path.join(fixtures_dir, "test_fretboard.jpg")
    no_fretboard_path = os.path.join(fixtures_dir, "test_no_fretboard.jpg")
    no_hand_path = os.path.join(fixtures_dir, "test_no_hand.jpg")

    create_fretboard_image(output_path=fretboard_path)
    create_no_fretboard_image(output_path=no_fretboard_path)
    create_no_hand_image(output_path=no_hand_path)

    print(f"Created: {fretboard_path}")
    print(f"Created: {no_fretboard_path}")
    print(f"Created: {no_hand_path}")
