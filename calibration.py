import cv2
import numpy as np
import glob

# Número de cantos internos
pattern_size = (7, 7)

# Tamanho de cada quadrado em cm
square_size = 3.0

# Pontos 3D conhecidos
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
objp *= square_size


objpoints = []  # pontos 3D
imgpoints = []  # pontos 2D

images = glob.glob("images/*.jpeg")

# Critério para refinamento dos cantos
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Detectar cantos
for filename in images:

    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
		gray,
		pattern_size,
		None
	)

    if found:

        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"[OK] {filename}")

        cv2.drawChessboardCorners(
			img,
			pattern_size,
			corners,
			found
		)

        cv2.imwrite(
			f"detected_{filename.split('/')[-1]}",
			img
		)

    else:
        print(f"[FALHOU] {filename}")

# Calibração
image_size = gray.shape[::-1]

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n===== RESULTADOS =====")

print("\nErro RMS:")
print(ret)

print("\nMatriz da câmera:")
print(camera_matrix)

print("\nCoeficientes de distorção:")
print(dist_coeffs)

print("\nNúmero de imagens utilizadas:")
print(len(objpoints))

np.savez(
    "calibration_data.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=np.array(rvecs),
    tvecs=np.array(tvecs),
    image_size=np.array(image_size),
)
