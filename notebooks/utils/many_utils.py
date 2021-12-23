import cv2
from IPython.display import HTML, FileLink, ProgressBar, clear_output, display
import numpy as np
from pathlib import Path

def create_superresolution_comparison_video(image, image_super, output_dir, image_name):
    image_bicubic = cv2.resize(image, tuple(image_super.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)
    FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
    result_video_path = output_dir / f"{image_name}_comparison.avi"
    video_target_height, video_target_width = (
        image_super.shape[0] // 2,
        image_super.shape[1] // 2,
    )

    out_video = cv2.VideoWriter(
        str(result_video_path),
        FOURCC,
        90,
        (video_target_width, video_target_height),
    )

    resized_result_image = cv2.resize(image_super, (video_target_width, video_target_height))[
        :, :, (2, 1, 0)
    ]
    resized_bicubic_image = cv2.resize(image_bicubic, (video_target_width, video_target_height))[
        :, :, (2, 1, 0)
    ]

    progress_bar = ProgressBar(total=video_target_width)
    progress_bar.display()

    for i in range(2, video_target_width):
        # Create a frame where the left part (until i pixels width) contains the
        # superresolution image, and the right part (from i pixels width) contains
        # the bicubic image
        comparison_frame = np.hstack(
            (
                resized_result_image[:, :i, :],
                resized_bicubic_image[:, i:, :],
            )
        )

        # create a small black border line between the superresolution
        # and bicubic part of the image
        comparison_frame[:, i - 1 : i + 1, :] = 0
        out_video.write(comparison_frame)
        progress_bar.progress = i
        progress_bar.update()
    out_video.release()
    clear_output()

    video_link = FileLink(result_video_path)
    video_link.html_link_str = "<a href='%s' download>%s</a>"
    display(HTML(f"The video has been saved to {video_link._repr_html_()}"))