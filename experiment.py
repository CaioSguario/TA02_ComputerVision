import cv2
import numpy as np

data = np.load("calibration_data.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]
rvecs = data["rvecs"]
tvecs = data["tvecs"]
successful_images = data["successful_images"]

square_size = 3.0
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
objp *= square_size

pattern_size = (7, 7)

for i, filename in enumerate(successful_images):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern_size,
        None
    )

    projected, _ = cv2.projectPoints(
        objp,
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    projected = projected.reshape(-1, 2)
    observed = corners.reshape(-1, 2)

    errors = np.linalg.norm(projected - observed, axis=1)

    print(filename)
    print("Erro médio:")
    print(errors.mean(), "pixels")

    print("\nMaior erro:")
    print(errors.max(), "pixels")

    print("\nMenor erro:")
    print(errors.min(), "pixels")

    print("\nErro dos 49 pontos:")
    print(errors)

    for point in projected:
        x, y = point.astype(int)

        cv2.circle(
            image,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

    output_name = "projection_" + filename.split("/")[-1]

    cv2.imwrite(
        output_name,
        image
    )

    print("\nImagem salva como", output_name)
