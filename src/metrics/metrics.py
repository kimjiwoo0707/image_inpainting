import cv2
import numpy as np
from skimage.metrics import structural_similarity as ski_ssim

def get_ssim_score(true: np.ndarray, pred: np.ndarray) -> float:
    return ski_ssim(true, pred, channel_axis=-1, data_range=pred.max() - pred.min())

def get_masked_ssim_score(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    true_masked = true[mask > 0]
    pred_masked = pred[mask > 0]
    return ski_ssim(true_masked, pred_masked, channel_axis=-1, data_range=pred.max() - pred.min())

def get_histogram_similarity(true: np.ndarray, pred: np.ndarray, cvt_color=cv2.COLOR_RGB2HSV) -> float:
    true_hsv = cv2.cvtColor(true, cvt_color)
    pred_hsv = cv2.cvtColor(pred, cvt_color)

    hist_true = cv2.calcHist([true_hsv], [0], None, [180], [0, 180])
    hist_pred = cv2.calcHist([pred_hsv], [0], None, [180], [0, 180])
    hist_true = cv2.normalize(hist_true, hist_true).flatten()
    hist_pred = cv2.normalize(hist_pred, hist_pred).flatten()

    return cv2.compareHist(hist_true, hist_pred, cv2.HISTCMP_CORREL)
