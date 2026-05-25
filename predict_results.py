import argparse
import pickle
from pathlib import Path

import cv2
import torch
from torch.backends import cudnn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import HybridNetsBackbone
from utils.constants import BINARY_MODE, MULTICLASS_MODE
from utils.utils import BBoxTransform, ClipBoxes, Params, letterbox, postprocess, scale_coords


def get_args():
    parser = argparse.ArgumentParser('HybridNets prediction result exporter')
    parser.add_argument('-p', '--project', type=str, default='waterscenes_imagenet',
                        help='Project yaml name in projects/')
    parser.add_argument('-bb', '--backbone', type=str, default=None,
                        help='Backbone name used when training, if any')
    parser.add_argument('-c', '--compound_coef', type=int, default=3,
                        help='Coefficient of efficientnet backbone')
    parser.add_argument('-w', '--load_weights', type=str, default='/data/hybridnet/weights/hybridnets_weight.pth',
                        help='Pretrained weight/checkpoint path')
    parser.add_argument('--split_file', type=str, default='/data_ssd/datasets/WaterScenes/MIPC_shipOnly/2007_test.txt',
                        help='Test split txt. The first token of each line is treated as image path/name')
    parser.add_argument('--data_path', type=str, default='/data_ssd/datasets/WaterScenes',
                        help='Dataset root path')
    parser.add_argument('--image_folder', type=str, default='images',
                        help='Image folder name under dataset root')
    parser.add_argument('--output_root', type=str, default='/data/hybridnet/predicted_results',
                        help='Root directory for exported predictions')
    parser.add_argument('--conf_thresh', type=float, default=0.35,
                        help='Detection confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.35,
                        help='NMS IoU threshold')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Reserved for interface compatibility; this script reads images directly')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--float16', action='store_true',
                        help='Use FP16 inference on CUDA')
    parser.add_argument('--voc_classes', type=int, default=9,
                        help='Number of classes written in VOC segmentation PNG, including background')
    return parser.parse_args()


def load_state_dict(weights_path, device):
    try:
        checkpoint = torch.load(weights_path, map_location=device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    return state_dict


def infer_seg_channels(state_dict):
    for key in ('segmentation_head.0.weight', 'segmentation_head.0.conv.weight'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key, value in state_dict.items():
        if key.startswith('segmentation_head') and key.endswith('weight') and value.ndim == 4:
            return int(value.shape[0])
    raise KeyError('Cannot infer segmentation output channels from weights.')


def read_split_images(split_file, data_root, image_folder):
    data_root = Path(data_root)
    image_root = data_root / image_folder
    suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []

    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        token = line.split()[0]
        path = Path(token)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([data_root / token, image_root / token])
            if path.suffix:
                candidates.append(image_root / path.name)
            else:
                candidates.extend(image_root / f'{token}{suffix}' for suffix in suffixes)
                candidates.extend(data_root / f'{token}{suffix}' for suffix in suffixes)

        resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        image_paths.append(resolved)

    return image_paths


def preprocess_image(image_path, resized_shape, transform):
    ori_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if ori_img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    h0, w0 = ori_img.shape[:2]
    input_size = max(resized_shape) if isinstance(resized_shape, list) else resized_shape
    ratio_resize = input_size / max(h0, w0)
    resized = cv2.resize(
        ori_img,
        (int(w0 * ratio_resize), int(h0 * ratio_resize)),
        interpolation=cv2.INTER_AREA if ratio_resize < 1 else cv2.INTER_LINEAR,
    )
    h, w = resized.shape[:2]
    (letterboxed, _), _, pad = letterbox((resized, None), input_size, auto=True, scaleup=False)
    shapes = ((h0, w0), ((h / h0, w / w0), pad))
    tensor = transform(letterboxed)
    return ori_img, tensor, shapes


def save_detection_txt(output_path, prediction, obj_list):
    with open(output_path, 'w') as f:
        for box, cls_id, score in zip(prediction['rois'], prediction['class_ids'], prediction['scores']):
            x1, y1, x2, y2 = box.tolist()
            cls_id = int(cls_id)
            cls_name = obj_list[cls_id] if cls_id < len(obj_list) else str(cls_id)
            f.write(f'{cls_name} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')


def segmentation_to_voc_mask(segmentation, shape, voc_classes):
    if segmentation.shape[1] == 1:
        mask = torch.where(segmentation[:, 0] >= 0, 1, 0)
    else:
        mask = torch.argmax(segmentation, dim=1)

    mask = mask[0].detach().cpu().numpy().astype(np.uint8)
    pad_h = int(shape[1][1][1])
    pad_w = int(shape[1][1][0])
    if pad_h > 0:
        mask = mask[pad_h:-pad_h, :]
    if pad_w > 0:
        mask = mask[:, pad_w:-pad_w]
    mask = cv2.resize(mask, dsize=shape[0][::-1], interpolation=cv2.INTER_NEAREST)
    return np.clip(mask, 0, voc_classes - 1).astype(np.uint8)


@torch.no_grad()
def main():
    args = get_args()
    params = Params(f'projects/{args.project}.yml')
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    cudnn.benchmark = use_cuda

    output_root = Path(args.output_root)
    det_dir = output_root / 'DetectionResults'
    seg_dir = output_root / 'SegmentationClass'
    det_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_state_dict(args.load_weights, device)
    seg_channels = infer_seg_channels(state_dict)
    seg_mode = BINARY_MODE if seg_channels == 1 else MULTICLASS_MODE
    seg_classes = 1 if seg_mode == BINARY_MODE else seg_channels - 1

    model = HybridNetsBackbone(
        num_classes=len(params.obj_list),
        compound_coef=args.compound_coef,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
        seg_classes=seg_classes,
        backbone_name=args.backbone,
        seg_mode=seg_mode,
        pretrained_backbone=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.requires_grad_(False)
    model.eval().to(device)
    if use_cuda and args.float16:
        model.half()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=params.mean, std=params.std),
    ])
    image_paths = read_split_images(args.split_file, args.data_path, args.image_folder)
    regress_boxes = BBoxTransform()
    clip_boxes = ClipBoxes()

    for image_path in tqdm(image_paths, ascii=True):
        ori_img, tensor, shape = preprocess_image(image_path, params.model['image_size'], transform)
        tensor = tensor.unsqueeze(0).to(device)
        tensor = tensor.to(torch.float16 if use_cuda and args.float16 else torch.float32)

        _, regression, classification, anchors, segmentation = model(tensor)
        predictions = postprocess(
            tensor, anchors, regression, classification,
            regress_boxes, clip_boxes, args.conf_thresh, args.iou_thresh
        )
        predictions[0]['rois'] = scale_coords(tensor.shape[2:], predictions[0]['rois'], shape[0], shape[1])

        stem = image_path.stem
        save_detection_txt(det_dir / f'{stem}.txt', predictions[0], params.obj_list)
        voc_mask = segmentation_to_voc_mask(segmentation, shape, args.voc_classes)
        cv2.imwrite(str(seg_dir / f'{stem}.png'), voc_mask)

    print(f'Saved detection txt files to: {det_dir}')
    print(f'Saved VOC segmentation PNG files to: {seg_dir}')


if __name__ == '__main__':
    main()
