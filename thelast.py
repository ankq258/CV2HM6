import numpy as np
import cv2
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('PSMNet')
from PSMNet.models.stackhourglass import PSMNet

def load_psmnet(model_path):
    model = PSMNet(maxdisp=192)
    model = torch.nn.DataParallel(model).cuda()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    return model


def compute_psmnet(left_img, right_img, model):

    left = left_img.astype(np.float32) / 255.0
    right = right_img.astype(np.float32) / 255.0
    h, w = left.shape[:2]

    new_h = (h // 64) * 64
    new_w = (w // 64) * 64

    left = cv2.resize(left, (new_w, new_h))
    right = cv2.resize(right, (new_w, new_h))
    left_tensor = torch.from_numpy(left)\
        .permute(2,0,1)\
        .unsqueeze(0)\
        .float()\
        .cuda()
    right_tensor = torch.from_numpy(right)\
        .permute(2,0,1)\
        .unsqueeze(0)\
        .float()\
        .cuda()

    with torch.no_grad():
        disparity = model(
            left_tensor,
            right_tensor
        )

    disparity = torch.squeeze(
        disparity
    ).cpu().numpy()
    disparity = cv2.resize(
        disparity,
        (w, h)
    )
    return disparity

pattern_size = (9, 6)
square_size = 35.0

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

objp = np.zeros((54, 3), np.float32)
objp[:, :2] = np.mgrid[
    0:9,
    0:6
].T.reshape(-1, 2) * square_size
left_imgs = sorted(
    Path("calib/left").glob("*.png")
)
right_imgs = sorted(
    Path("calib/right").glob("*.png")
)

objpoints = []
left_pts = []
right_pts = []
first_img_shape = None
for l, r in zip(left_imgs, right_imgs):
    gl = cv2.cvtColor(
        cv2.imread(str(l)),
        cv2.COLOR_BGR2GRAY
    )
    gr = cv2.cvtColor(
        cv2.imread(str(r)),
        cv2.COLOR_BGR2GRAY
    )

    if first_img_shape is None:
        first_img_shape = gl.shape[::-1]
    ret_l, cl = cv2.findChessboardCorners(
        gl,
        pattern_size,
        None
    )
    ret_r, cr = cv2.findChessboardCorners(
        gr,
        pattern_size,
        None
    )

    if ret_l and ret_r:
        objpoints.append(objp)
        left_pts.append(
            cv2.cornerSubPix(
                gl,
                cl,
                (11,11),
                (-1,-1),
                criteria
            )
        )
        right_pts.append(
            cv2.cornerSubPix(
                gr,
                cr,
                (11,11),
                (-1,-1),
                criteria
            )
        )

ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
    objpoints,
    left_pts,
    gl.shape[::-1],
    None,
    None
)
ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
    objpoints,
    right_pts,
    gr.shape[::-1],
    None,
    None
)
ret, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    left_pts,
    right_pts,
    K_l,
    D_l,
    K_r,
    D_r,
    gl.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_l,
    D_l,
    K_r,
    D_r,
    first_img_shape,
    R,
    T,
    alpha=0
)
map1x, map1y = cv2.initUndistortRectifyMap(
    K_l,
    D_l,
    R1,
    P1,
    first_img_shape,
    cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    K_r,
    D_r,
    R2,
    P2,
    first_img_shape,
    cv2.CV_32FC1
)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    blockSize=5,

    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,

    disp12MaxDiff=1,
    uniquenessRatio=10,

    speckleWindowSize=100,
    speckleRange=32,

    preFilterCap=63,

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print("Loading PSMNet...")

model_path = 'pretrained_model_KITTI2015.tar'
if not Path(model_path).exists():
    print(f"Ошибка: Файл модели '{model_path}' не найден!")
    sys.exit(1)

psmnet_model = load_psmnet(model_path)

print("PSMNet loaded")

video_path = 'video.mp4'
if not Path(video_path).exists():
    print(f"Ошибка: Видеофайл '{video_path}' не найден!")
    print(f"Текущая директория: {Path.cwd()}")
    print("Доступные файлы в директории:")
    for file in Path().glob("*"):
        print(f"  - {file}")
    sys.exit(1)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видеофайл '{video_path}'")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Видео успешно загружено:")
print(f"  Размер: {width}x{height}")
print(f"  FPS: {fps}")

split = width // 2
output_width = split * 2
output_height = height

output_path = output_dir / "result_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))

if not out.isOpened():
    print(f"Ошибка: Не удалось создать видеорекордер для '{output_path}'")
    sys.exit(1)


frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]
    split = w // 2

    left_img = frame[:, :split]
    right_img = frame[:, split:]
    rect_left = cv2.remap(
        left_img,
        map1x,
        map1y,
        cv2.INTER_LINEAR
    )

    rect_right = cv2.remap(
        right_img,
        map2x,
        map2y,
        cv2.INTER_LINEAR
    )

    gray_left = cv2.cvtColor(
        rect_left,
        cv2.COLOR_BGR2GRAY
    )
    gray_right = cv2.cvtColor(
        rect_right,
        cv2.COLOR_BGR2GRAY
    )
    disparity_sgbm = stereo.compute(
        gray_left,
        gray_right
    ).astype(np.float32) / 16.0

    disparity_sgbm[disparity_sgbm < 0] = 0

    disparity_sgbm_norm = cv2.normalize(
        disparity_sgbm,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    disparity_sgbm_color = cv2.applyColorMap(
        disparity_sgbm_norm.astype(np.uint8),
        cv2.COLORMAP_MAGMA
    )
    disparity_psm = compute_psmnet(
        rect_left,
        rect_right,
        psmnet_model
    )
    disparity_psm_norm = cv2.normalize(
        disparity_psm,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    disparity_psm_color = cv2.applyColorMap(
        disparity_psm_norm.astype(np.uint8),
        cv2.COLORMAP_MAGMA
    )
    cv2.putText(
        disparity_sgbm_color,
        "SGBM",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )
    cv2.putText(
        disparity_psm_color,
        "PSMNet",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    combined = np.hstack([
        disparity_sgbm_color,
        disparity_psm_color
    ])

    out.write(combined)
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed_time = time.time() - start_time
        fps_processing = frame_count / elapsed_time
        print(f"Обработано кадров: {frame_count}/{total_frames} "
              f"({frame_count/total_frames*100:.1f}%) "
              f"Скорость: {fps_processing:.1f} fps")

cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time
