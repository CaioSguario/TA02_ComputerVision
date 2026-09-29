import cv2
import numpy as np

data = np.load("calibration_data.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]
successful_images = data["successful_images"]

for filename in successful_images:
    img = cv2.imread(filename)

    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        0,
        (w, h)
    )

    undistorted = cv2.undistort(
        img,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    output_name = "undistorted_" + filename.split("/")[-1]

    cv2.imwrite(
        output_name,
        undistorted
    )

    print("Imagem corrigida salva como", output_name)
