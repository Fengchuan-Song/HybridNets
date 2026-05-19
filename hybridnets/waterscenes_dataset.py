from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm

from hybridnets.dataset import BddDataset
from utils.constants import MULTILABEL_MODE


class WaterScenesDataset(BddDataset):
    """WaterScenes adapter for split image lists plus xyxy label txt files."""

    def __init__(self, params, is_train, inputsize=[320, 320], transform=None,
                 seg_mode=MULTILABEL_MODE, debug=False, split=None):
        self.is_train = is_train
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        self.seg_list = params.seg_list
        self.dataset = params.dataset
        self.obj_list = params.obj_list
        self.obj_combine = params.obj_combine
        self.traffic_light_color = False
        self.seg_mode = seg_mode
        self.shapes = np.array(params.dataset['org_img_size'])
        self.mosaic_border = [-1 * self.inputsize[1] // 2, -1 * self.inputsize[0] // 2]

        self.root = Path(params.dataset['dataroot'])
        self.img_root = Path(params.dataset.get('image_root', self.root / 'images'))
        self.label_root = Path(params.dataset.get('label_root', self.root / 'labels'))
        self.semantic_root = Path(params.dataset['semantic_root'])
        self.waterline_root = Path(params.dataset['waterline_root'])

        split_key = split if split is not None else ('train_set' if is_train else 'test_set')
        split_path = Path(params.dataset[split_key])
        if not split_path.is_absolute():
            split_path = self.root / split_path
        self.split_path = split_path

        self.db = self._get_db()
        if debug:
            self.db = self.db[:50]

    def _get_db(self):
        print(f'building WaterScenes database from {self.split_path}...')
        gt_db = []
        with open(self.split_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in tqdm(lines, ascii=True):
            image_path, boxes = self._parse_split_line(line)
            stem = Path(image_path).stem
            rec = {
                'image': str(image_path),
                'label': boxes,
                'drivable': str(self.semantic_root / f'{stem}.png'),
                'waterline': str(self.waterline_root / f'{stem}.png'),
            }
            gt_db.append(rec)
        print('WaterScenes database build finish')
        return gt_db

    def _parse_split_line(self, line):
        parts = line.split()
        image_path = self._resolve_image_path(parts[0])
        raw_boxes = parts[1:]

        boxes = []
        if raw_boxes:
            boxes = self._parse_box_tokens(raw_boxes)
        else:
            boxes = self._read_label_file(image_path)

        if boxes:
            gt = np.array(boxes, dtype=np.float32)
        else:
            gt = np.zeros((0, 5), dtype=np.float32)
        return image_path, gt

    def _parse_box_tokens(self, raw_boxes):
        boxes = []
        comma_boxes = [token for token in raw_boxes if ',' in token]
        if comma_boxes:
            for token in comma_boxes:
                vals = [v for v in token.split(',') if v != '']
                self._append_box(vals, boxes)
        else:
            for i in range(0, len(raw_boxes), 5):
                self._append_box(raw_boxes[i:i + 5], boxes)
        return boxes

    def _read_label_file(self, image_path):
        label_path = self._resolve_label_path(image_path)
        if label_path is None:
            return []

        boxes = []
        img_h, img_w = self.shapes
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.replace(',', ' ').split()
                self._append_yolo_box(tokens, boxes, img_w, img_h)
        return boxes

    def _resolve_label_path(self, image_path):
        image_path = Path(image_path)
        rel_candidates = []
        try:
            rel_candidates.append(image_path.relative_to(self.img_root))
        except ValueError:
            pass
        rel_candidates.append(Path(image_path.name))

        candidates = []
        for rel in rel_candidates:
            candidates.append((self.label_root / rel).with_suffix('.txt'))
        candidates.append(self.label_root / f'{image_path.stem}.txt')

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _append_box(self, vals, boxes):
        if len(vals) < 4:
            return
        if len(vals) >= 5 and self._looks_like_class_id(vals[0]) and not self._looks_like_class_id(vals[4]):
            cls_id = int(float(vals[0]))
            x1, y1, x2, y2 = [float(v) for v in vals[1:5]]
        else:
            x1, y1, x2, y2 = [float(v) for v in vals[:4]]
            cls_id = int(float(vals[4])) if len(vals) > 4 else 0
        if cls_id != 0:
            return
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 > x1 and y2 > y1:
            boxes.append([0, x1, y1, x2, y2])

    def _append_yolo_box(self, vals, boxes, img_w, img_h):
        if len(vals) < 5:
            return
        cls_id = int(float(vals[0]))
        if cls_id != 0:
            return

        cx, cy, w, h = [float(v) for v in vals[1:5]]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 > x1 and y2 > y1:
            boxes.append([0, x1, y1, x2, y2])

    def _looks_like_class_id(self, value):
        try:
            number = float(value)
        except ValueError:
            return False
        return number.is_integer() and 0 <= int(number) < len(self.obj_list)

    def _resolve_image_path(self, token):
        token = token.strip()
        path = Path(token)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([
                self.root / token,
                self.img_root / token,
                self.img_root / f'{token}.jpg' if path.suffix == '' else self.img_root / path.name,
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    @staticmethod
    def _read_mask(path, shape, kind):
        mask = cv2.imread(str(path), 0)
        if mask is None:
            return np.zeros(shape, dtype=np.uint8)
        if kind == 'drivable':
            mask = np.where((mask == 19) | (mask == 8), 255, 0).astype(np.uint8)
        else:
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        return mask

    def load_image(self, index):
        data = self.db[index]
        det_label = data['label']
        img = cv2.imread(data['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(f"Image not found: {data['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]
        seg_label = OrderedDict()
        seg_label['drivable'] = self._read_mask(data['drivable'], (h0, w0), 'drivable')
        seg_label['waterline'] = self._read_mask(data['waterline'], (h0, w0), 'waterline')

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        r = resized_shape / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            for seg_class in seg_label:
                seg_label[seg_class] = cv2.resize(
                    seg_label[seg_class], (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]

        labels = []
        if det_label.size > 0:
            labels = det_label.copy()
            labels[:, [1, 3]] = labels[:, [1, 3]] / w0 * w
            labels[:, [2, 4]] = labels[:, [2, 4]] / h0 * h

        return img, labels, seg_label, (h0, w0), (h, w), data['image']
