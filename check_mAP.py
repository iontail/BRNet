import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image
import random
import torch

seed = 18412
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_predictions(pred_file):
    """Read predictions from a file."""
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            bbox = list(map(float, parts[:4]))
            score = float(parts[4])
            predictions.append((bbox, score))
    return sorted(predictions, key=lambda x: x[1], reverse=True)

def read_ground_truth(gt_file):
    """Read ground truth from a file."""
    ground_truths = []
    with open(gt_file, 'r') as f:
        for line_idx, line in enumerate(f):
            # 첫 줄은 무시 (필요한 경우에만)
            if line_idx == 0 and not line.strip()[0].isdigit():
                continue
            parts = line.strip().split()
            if len(parts) != 4:  # [x1, y1, x2, y2] 형식이 아닌 경우 무시
                continue
            bbox = list(map(float, parts))
            ground_truths.append(bbox)
    return ground_truths

def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def evaluate_map(gt_boxes, pred_list, iou_threshold=0.5):
    """
    gt_boxes: numpy array of shape (N_gt, 4)
    pred_list: list of tuples (box, score), where box = [x1, y1, x2, y2]
    """
    # Convert list of (box, score) into numpy array
    pred_boxes = np.array([box + [score] for box, score in pred_list])

    # Sort by score descending
    pred_boxes = pred_boxes[pred_boxes[:, 4].argsort()[::-1]]
    matched_gt = set()
    tp = []
    fp = []

    for pred in pred_boxes:
        box_pred = pred[:4]
        score = pred[4]

        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(box_pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.array(tp)
    fp = np.array(fp)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / len(gt_boxes)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

    # AP: trapezoidal rule
    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap

def calculate_map_for_single_file(pred_file, gt_file, iou_threshold=0.5):
    """Calculate AP for a single prediction and ground truth file."""
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return 0
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        return 0

    predictions = read_predictions(pred_file)
    ground_truths = read_ground_truth(gt_file)
    
    print(f"Found {len(predictions)} predictions and {len(ground_truths)} ground truths")
    
    ap = evaluate_map(ground_truths, predictions, iou_threshold)
    print(f"Average Precision: {ap:.4f}")
    
    return ap

def visualize_detections(image_path, predictions, ground_truths, iou_threshold=0.5):
    """
    이미지에 예측값과 GT를 시각화하는 함수
    - 파란색: Ground Truth
    - 빨간색: 예측값 (매칭되지 않은 False Positive)
    - 초록색: 매칭된 예측값 (True Positive)
    """
    try:
        # 이미지 불러오기
        img = Image.open(image_path)
        img_np = np.array(img)
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {e}")
        img_width, img_height = max(ground_truths, key=lambda x: x[2])[2], max(ground_truths, key=lambda x: x[3])[3]
        # int형으로 변환
        img_width, img_height = int(img_width), int(img_height)
        img_np = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 이미지 크기 확인
    height, width = img_np.shape[:2]
    
    # Figure 생성
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    
    # Ground Truth 그리기 (파란색)
    for i, gt_bbox in enumerate(ground_truths):
        x1, y1, x2, y2 = gt_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'GT #{i}', color='blue', fontsize=2, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # 각 예측과 GT 간의 IoU 계산 및 최적 매칭 찾기
    gt_matched = [False] * len(ground_truths)
    matched_pairs = []
    
    # 신뢰도 점수 순으로 예측값 처리
    for pred_idx, (pred_bbox, score) in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_bbox in enumerate(ground_truths):
            if not gt_matched[gt_idx]:
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # IoU가 임계값을 넘으면 매칭
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            matched_pairs.append((pred_idx, best_gt_idx, best_iou))
    
    # 예측값 그리기 (매칭된 것은 초록색, 매칭 안된 것은 빨간색)
    for pred_idx, (pred_bbox, score) in enumerate(predictions):
        x1, y1, x2, y2 = pred_bbox
        
        # 매칭 여부 확인
        matched = False
        matched_gt = -1
        iou_value = 0
        
        for p_idx, gt_idx, iou in matched_pairs:
            if p_idx == pred_idx:
                matched = True
                matched_gt = gt_idx
                iou_value = iou
                break
        
        # 박스 그리기
        color = 'green' if matched else 'red'

            
        # 텍스트 정보 추가
        if matched:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y2+10, f'Pred #{pred_idx}, Score: {score:.2f}\nMatch: GT #{matched_gt}, IoU: {iou_value:.2f}', 
                  color=color, fontsize=2, bbox=dict(facecolor='white', alpha=0.7))
        else:
            """
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y2+10, f'Pred #{pred_idx}, Score: {score:.2f}\nFP', 
                  color=color, fontsize=2, bbox=dict(facecolor='white', alpha=0.7))
            """
            # 매칭되지 않은 예측값은 그리지 않음
            continue
    # 미검출된 GT 표시
    fn_count = sum(1 for matched in gt_matched if not matched)
    tp_count = len(matched_pairs)
    fp_count = len(predictions) - tp_count
    
    # 제목에 요약 정보 추가
    ax.set_title(f'Object Detection Results\nTP: {tp_count}, FP: {fp_count}, FN: {fn_count}')
    
    # 축 설정
    ax.set_axis_off()
    plt.tight_layout()
    
    return fig

def pred(index, result_path=None, iou_threshold=0.5):
    pred_file = os.path.join(result_path, f"{index}_ZSDA.txt")
    gt_file = os.path.join("dataset/DarkFace/DarkFace_val/label", f"{index}.txt") ### GT label
    image_file = os.path.join("dataset/DarkFace/DarkFace_val/image", f"{index}.png") ### GT image

    ap = calculate_map_for_single_file(pred_file, gt_file, iou_threshold)
    print(f"AP for {index}: {ap:.4f}")

    # 예측값과 GT 불러오기
    predictions = read_predictions(pred_file)
    ground_truths = read_ground_truth(gt_file)
    '''
    # 시각화
    fig = visualize_detections(image_file, predictions, ground_truths, iou_threshold)

    # 시각화 결과 저장
    save_path = result_path + f"/visualization_iou{iou_threshold*100}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, f"{index}_visualization.png"), dpi=300, bbox_inches='tight')
    '''
    return ap
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate and visualize object detection results.")
    parser.add_argument('--result_path', type=str, default=None, help='Path to the result files')
    parser.add_argument('--index', type=str, default=None, help='Index of the file to evaluate')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for evaluation')
    args = parser.parse_args()

    if args.result_path:
        preds = os.listdir(args.result_path)
        # _ZSDA.txt 앞에 있는 숫자만 들고오기
        indexs = [int(pred.split('_')[0]) for pred in preds if pred.endswith('_ZSDA.txt')]
        indexs.sort()
    else:
        indexs = [args.index]

    mAP = 0

    for index in indexs:
        print(f"Processing index: {index}")
        ap = pred(index, args.result_path, args.iou)
        mAP += ap
    mAP /= len(indexs)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print("Evaluation complete.")
