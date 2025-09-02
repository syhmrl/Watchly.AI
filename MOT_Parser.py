from collections import defaultdict
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import numpy as np
import motmetrics as mm
import os
import csv
import cv2

def load_and_clean_mot_txt(path, src_size=None, dst_size=None, save_clean_path=None, min_area=10):
    """
    Read a CVAT MOT 1.1 txt and return a dict: frame -> list of (id, [x1,y1,x2,y2]).
    If src_size provided (w,h) and dst_size provided (w,h), it will scale boxes from src -> dst.
    If save_clean_path provided, will write a cleaned MOT txt with columns: frame,id,left,top,width,height
    """
    gt_by_frame = defaultdict(list)

    def scale_box(box, src_size, dst_size):
        sx = dst_size[0] / src_size[0]
        sy = dst_size[1] / src_size[1]
        x1, y1, x2, y2 = box
        return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # take first 6 columns; some exports include extra columns after
            if len(row) < 6:
                continue
            try:
                frame = int(float(row[0]))
                obj_id = int(float(row[1]))
                left = float(row[2])
                top  = float(row[3])
                width = float(row[4])
                height = float(row[5])
            except Exception as e:
                # skip malformed row
                continue

            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height

            # optional scale from src -> dst
            if src_size is not None and dst_size is not None:
                x1,y1,x2,y2 = scale_box((x1,y1,x2,y2), src_size, dst_size)

            # optional clipping (if dst_size provided, clip to dst)
            if dst_size is not None:
                w,h = dst_size
                x1 = max(0, min(x1, w-1))
                x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1))
                y2 = max(0, min(y2, h-1))

            # sanity: require positive area
            if (x2 - x1) * (y2 - y1) < min_area:
                continue

            gt_by_frame[frame].append((obj_id, [x1, y1, x2, y2]))

    # save cleaned MOT txt if requested (frame,id,left,top,width,height)
    if save_clean_path:
        os.makedirs(os.path.dirname(save_clean_path) or '.', exist_ok=True)
        with open(save_clean_path, 'w', newline='') as out:
            writer = csv.writer(out)
            for frame in sorted(gt_by_frame.keys()):
                for obj_id, box in gt_by_frame[frame]:
                    x1,y1,x2,y2 = box
                    left = x1
                    top  = y1
                    width = x2 - x1
                    height = y2 - y1
                    writer.writerow([frame, obj_id, left, top, width, height])

    return gt_by_frame

def load_preds_csv(path):
    preds = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            frame = int(float(r['frame']))
            pid = int(float(r['id']))
            x1 = float(r['x1']); y1 = float(r['y1']); x2 = float(r['x2']); y2 = float(r['y2'])
            score = float(r.get('score', 1.0))
            preds[frame].append((pid, [x1,y1,x2,y2], score))
    return preds

def load_mot_txt(path):
    """
    Returns dict: frame -> list of (id, [x1,y1,x2,y2])
    Assumes columns: frame,id,left,top,width,height, ... (left/top/width/height in pixels)
    """
    out = defaultdict(list)
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            frame = int(float(row[0]))
            obj_id = int(float(row[1]))
            left = float(row[2]); top = float(row[3])
            width = float(row[4]); height = float(row[5])
            x1 = left; y1 = top; x2 = left + width; y2 = top + height
            out[frame].append((obj_id, [x1,y1,x2,y2]))
    return out

def get_video_dimensions(video_path):
    """
    Retrieves the width and height of a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing (width, height) of the video,
                or (None, None) if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release() # Release the video capture object
    return width, height

def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def compute_annotation_stats(gt_by_frame, dst_size=None):
    # Returns dictionary of stats to flag issues
    total_boxes = 0
    clipped_count = 0
    small_count = 0
    zero_area = 0
    widths = []
    heights = []
    areas = []
    frames = sorted(gt_by_frame.keys())
    for f in frames:
        for obj_id, box in gt_by_frame[f]:
            total_boxes += 1
            x1,y1,x2,y2 = box
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                zero_area += 1
            if w*h < 100:  # tiny area threshold, tune as needed
                small_count += 1
            widths.append(w); heights.append(h); areas.append(w*h)
            if dst_size:
                W,H = dst_size
                if x1 < 0 or y1 < 0 or x2 > W-1 or y2 > H-1:
                    clipped_count += 1
    stats = {
        'total_boxes': total_boxes,
        'zero_area': zero_area,
        'tiny_boxes(<100px)': small_count,
        'clipped_boxes': clipped_count,
        'avg_width': np.mean(widths) if widths else 0,
        'avg_height': np.mean(heights) if heights else 0,
        'median_area': np.median(areas) if areas else 0,
        'frame_min': min(frames) if frames else None,
        'frame_max': max(frames) if frames else None
    }
    return stats

def overlay_and_save_samples(video_path, gt_by_frame, dst_size, out_dir='gt_overlays', sample_every_n=30, max_samples=50):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frames: {total}")
    frames_to_save = []
    # choose sample frames: uniform and where gt exists
    for f in sorted(gt_by_frame.keys()):
        frames_to_save.append(f)
    # downsample frames_to_save
    frames_to_save = frames_to_save[::sample_every_n][:max_samples]
    # ensure unique and valid
    frames_to_save = [int(min(max(0, f-1), total-1)) for f in frames_to_save]  # convert CVAT 1-based -> 0-based
    saved = 0
    for idx in sorted(set(frames_to_save)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # if you processed in dst_size, resize frame to dst_size for comparison
        if dst_size:
            frame = cv2.resize(frame, (dst_size[0], dst_size[1]))
        fnum_for_lookup = idx + 1  # because GT likely uses 1-based frame numbers
        for obj_id, box in gt_by_frame.get(fnum_for_lookup, []):
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, str(obj_id), (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out_path = os.path.join(out_dir, f'frame_{idx:06d}.jpg')
        cv2.imwrite(out_path, frame)
        saved += 1
    cap.release()
    print(f"Saved {saved} overlay images to {out_dir}")
    
def evaluate_detection_metrics(gt_by_frame, pred_by_frame, iou_thr=0.5):
    total_TP = total_FP = total_FN = 0
    per_frame = {}
    frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    for f in frames:
        gts = [b for _, b in gt_by_frame.get(f, [])]
        preds = [b for _, b, _ in pred_by_frame.get(f, [])]

        if len(gts)==0 and len(preds)==0:
            per_frame[f] = {'TP':0,'FP':0,'FN':0}
            continue

        if len(gts)==0:
            per_frame[f] = {'TP':0,'FP':len(preds),'FN':0}
            total_FP += len(preds)
            continue
        if len(preds)==0:
            per_frame[f] = {'TP':0,'FP':0,'FN':len(gts)}
            total_FN += len(gts)
            continue

        # cost = 1 - IoU, Hungarian solves min cost
        cost = np.ones((len(gts), len(preds)), dtype=float)
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                cost[i,j] = 1.0 - iou(g,p)
        # forbid matches with IoU < iou_thr by setting high cost
        cost[cost > (1.0 - iou_thr)] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r,c in zip(row_ind, col_ind):
            if cost[r,c] < 1e5:
                matches.append((r,c))

        TP = len(matches)
        FP = len(preds) - TP
        FN = len(gts) - TP

        total_TP += TP; total_FP += FP; total_FN += FN
        per_frame[f] = {'TP':TP,'FP':FP,'FN':FN}

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP)>0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN)>0 else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall)>0 else 0.0

    summary = {
        'TP': total_TP, 'FP': total_FP, 'FN': total_FN,
        'precision': precision, 'recall': recall, 'f1': f1,
        'frames_evaluated': len(per_frame)
    }
    return summary, per_frame

def build_motmetrics_acc(gt_by_frame, pred_by_frame, iou_threshold=0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    # frames union
    frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    for f in frames:
        gt_items = gt_by_frame.get(f, [])
        trk_items = pred_by_frame.get(f, [])
        gt_ids = [str(x[0]) for x in gt_items]
        gt_boxes = [x[1] for x in gt_items]
        trk_ids = [str(x[0]) for x in trk_items]
        trk_boxes = [x[1] for x in trk_items]

        if len(gt_boxes)==0 and len(trk_boxes)==0:
            # must still call update with empty lists
            acc.update([], [], np.empty((0,0)))
            continue

        # compute distance matrix = 1 - IoU
        if len(gt_boxes)>0 and len(trk_boxes)>0:
            D = np.zeros((len(gt_boxes), len(trk_boxes)), dtype=float)
            for i,g in enumerate(gt_boxes):
                for j,t in enumerate(trk_boxes):
                    # compute IoU
                    ix1 = max(g[0], t[0]); iy1 = max(g[1], t[1])
                    ix2 = min(g[2], t[2]); iy2 = min(g[3], t[3])
                    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                    inter = iw*ih
                    ga = max(0, g[2]-g[0]) * max(0, g[3]-g[1])
                    ta = max(0, t[2]-t[0]) * max(0, t[3]-t[1])
                    union = ga + ta - inter
                    iou = inter/union if union>0 else 0.0
                    D[i,j] = 1.0 - iou
            # forbid matches with IoU < threshold by giving huge cost
            D[D > (1.0 - iou_threshold)] = 1e6
        else:
            D = np.ones((len(gt_boxes), len(trk_boxes)), dtype=float)

        acc.update(gt_ids, trk_ids, D)
    return acc

def compute_and_print_metrics(acc):
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='eval')
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    return summary

if __name__ == "__main__":
    relative_path = os.path.join("..", "video_annotated_data/gt", "gt.txt")
    absolute_path = os.path.abspath(relative_path)
    
    if Path(absolute_path).is_file():
        print(f"The file '{absolute_path}' exists.")
    else:
        print(f"The file '{absolute_path}' does not exist or is not a regular file.")
        
    video_file = "video/test_4.mp4"
    width, height = get_video_dimensions(video_file)
    
    if width is not None and height is not None:
        print(f"Video Width: {width} pixels")
        print(f"Video Height: {height} pixels")    
        
    src_size = (width, height)
    dst_size = (1280, 720)
    
    gt_by_frame = load_and_clean_mot_txt(absolute_path,
                                    src_size=src_size,
                                    dst_size=dst_size)
    
    pred_by_frame = load_mot_txt('predictions/test_4.mp4_yolo11n.pt_reidnone_preds.txt')
    
    acc = build_motmetrics_acc(gt_by_frame, pred_by_frame, iou_threshold=0.5)
    summary = compute_and_print_metrics(acc)
    
    #summary, per_frame = evaluate_detection_metrics(gt_by_frame, pred_by_frame, iou_thr=0.5)
    #print(summary)
    
    # stats
    # stats = compute_annotation_stats(gt_by_frame, dst_size=dst_size)
    # print(stats)

    # create overlays (saves images)
    # overlay_and_save_samples(video_file, gt_by_frame, dst_size, out_dir='gt_overlays', sample_every_n=30)
