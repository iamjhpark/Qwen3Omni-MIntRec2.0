import os
import re
import csv
import logging
import numpy as np

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import INTENT_LABELS, OOD_LABEL

logger = logging.getLogger(__name__)


class MIntRecDialogueDataset(Dataset):
    """MIntRec2.0 multi-turn dialogue 데이터셋.

    각 아이템은 하나의 dialogue (여러 utterance)를 반환한다.
    """

    def __init__(self, dialogues, ood_label_id):
        self.dialogues = dialogues
        self.ood_label_id = ood_label_id

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, index):
        return self.dialogues[index]


def load_video_frames(face_dir, num_frames=8, face_size=(224, 224)):
    """Face ROI 이미지 폴더에서 균등 간격으로 샘플링하여 PIL Image 리스트로 반환.

    이미지가 num_frames 이하이면 전부 사용한다.
    Face ROI는 크기가 제각각이므로 face_size로 통일한다.
    """
    if face_dir is None or not os.path.isdir(face_dir):
        return None

    def _frame_sort_key(filename):
        """파일명에서 프레임 번호를 추출하여 숫자 기준으로 정렬."""
        m = re.search(r'_(\d+)_sim', filename)
        return int(m.group(1)) if m else 0

    image_files = sorted(
        (f for f in os.listdir(face_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))),
        key=_frame_sort_key,
    )
    if len(image_files) == 0:
        return None

    if len(image_files) <= num_frames:
        indices = list(range(len(image_files)))
    else:
        indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        img_path = os.path.join(face_dir, image_files[idx])
        try:
            frames.append(Image.open(img_path).convert('RGB').resize(face_size))
        except Exception as e:
            logger.warning(f'Failed to load image {img_path}: {e}')

    return frames if len(frames) > 0 else None


def read_dialogues(tsv_path, label_map, video_base_path):
    """TSV 파일을 파싱하여 dialogue 리스트를 반환한다.

    TSV 컬럼: Dialogue_id, Utterance_id, Text, Label,
              Start_timestamp, End_timestamp, Source, speaker_name
    """
    dialogues_dict = OrderedDict()

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if i == 0:
                continue

            dia_id = line[0]
            utt_id = line[1]
            text = line[2]
            label = line[3]
            speaker_name = line[7]

            label_id = label_map.get(label, label_map.get(OOD_LABEL, -1))

            face_dir_name = f'dia{dia_id}_utt{utt_id}'
            video_path = os.path.join(video_base_path, face_dir_name)

            if dia_id not in dialogues_dict:
                dialogues_dict[dia_id] = []

            dialogues_dict[dia_id].append({
                'text': text,
                'label_id': label_id,
                'speaker_name': speaker_name,
                'video_path': video_path,
                'dialogue_id': dia_id,
                'utterance_id': utt_id,
            })

    dialogues = []
    missing_videos = 0
    for dia_id, utterances in dialogues_dict.items():
        utterances.sort(key=lambda u: int(u['utterance_id']))
        for utt in utterances:
            if not os.path.isdir(utt['video_path']):
                missing_videos += 1
                utt['video_path'] = None
        dialogues.append({
            'dialogue_id': dia_id,
            'utterances': utterances,
        })

    if missing_videos > 0:
        logger.warning(f'{missing_videos} video files not found in {video_base_path}')

    return dialogues


def build_label_map():
    """Intent label → ID 매핑을 생성한다."""
    label_map = {label: i for i, label in enumerate(INTENT_LABELS)}
    label_map[OOD_LABEL] = len(INTENT_LABELS)
    return label_map


def prepare_data(args):
    """Train/Dev/Test 데이터셋과 DataLoader를 생성한다."""
    label_map = build_label_map()
    ood_label_id = len(INTENT_LABELS)

    datasets = {}
    num_train_inscope = 0

    for split in ['train', 'dev', 'test']:
        tsv_path = os.path.join(args.text_data_path, f'{split}.tsv')
        dialogues = read_dialogues(tsv_path, label_map, args.video_data_path)
        datasets[split] = MIntRecDialogueDataset(dialogues, ood_label_id)
        logger.info(f'[{split}] {len(dialogues)} dialogues loaded')

        if split == 'train':
            for d in dialogues:
                num_train_inscope += sum(
                    1 for u in d['utterances'] if u['label_id'] != ood_label_id
                )

    logger.info(f'In-scope training utterances: {num_train_inscope}')

    def collate_fn(batch):
        return batch

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=1, shuffle=True,
                            collate_fn=collate_fn, num_workers=0),
        'dev': DataLoader(datasets['dev'], batch_size=1, shuffle=False,
                          collate_fn=collate_fn, num_workers=0),
        'test': DataLoader(datasets['test'], batch_size=1, shuffle=False,
                           collate_fn=collate_fn, num_workers=0),
    }

    return dataloaders, num_train_inscope, ood_label_id
