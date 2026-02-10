# run.py (맨 위, 어떤 import보다 위에 두는 걸 권장)
import os
from pathlib import Path

def configure_hf_cache(cache_root: str | None = None):
    # 기본값: 레포 안쪽 큰 마운트 사용
    if cache_root is None:
        cache_root = "/home/work/Qwen3Omni-MIntRec2.0/_hf_cache"

    p = Path(cache_root)
    (p / "hub").mkdir(parents=True, exist_ok=True)
    (p / "transformers").mkdir(parents=True, exist_ok=True)
    (p / "datasets").mkdir(parents=True, exist_ok=True)
    (p / "tmp").mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(p)
    os.environ["HF_HUB_CACHE"] = str(p / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(p / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(p / "datasets")
    os.environ["TMPDIR"] = str(p / "tmp")

    # xet 비활성화 (xet_get 경로를 피하려고)
    os.environ["HF_HUB_DISABLE_XET"] = "1"

# configure_hf_cache()


import logging
import warnings
import datetime
import random
import numpy as np
import torch

from config import parse_args
from dataset import prepare_data
from trainer import Qwen3OmniTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(args, seed):
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_name = f'qwen3_omni_ind_seed{seed}_{time_str}'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fh = logging.FileHandler(os.path.join(args.log_dir, f'{log_name}.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


def load_model_and_processor(args):
    """Qwen3-Omni Thinker 모델을 로드하고 LoRA를 적용한다."""
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    logging.info(f'Loading processor from {args.model_name}...')
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model_name,
        min_pixels=128 * 28 * 28,     # 최소 비전 토큰 수 제한
        max_pixels=256 * 28 * 28,     # 최대 비전 토큰 수 제한 (OOM 방지)
    )

    logging.info(f'Loading model from {args.model_name}...')
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation='sdpa',
    )

    logging.info('Applying LoRA...')
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Fix F: LoRA가 적용된 모듈 확인 로깅
    lora_modules = [name for name, _ in model.named_modules() if 'lora' in name.lower()]
    logging.info(f'LoRA applied to {len(lora_modules)} modules')
    if len(lora_modules) == 0:
        logging.warning('WARNING: No LoRA modules found! Check target_modules configuration.')

    # gradient checkpointing과 use_cache=True는 충돌하므로 반드시 끈다
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, processor


def main():
    warnings.filterwarnings('ignore')
    args = parse_args()

    for seed in args.seeds:
        args.seed = seed
        logger = setup_logger(args, seed)
        set_seed(seed)

        logger.info(f"\n{'='*60}")
        logger.info(f"Seed: {seed}")
        logger.info(f"{'='*60}")

        # 시드별 출력 디렉토리
        seed_output_dir = os.path.join(args.output_dir, f'seed_{seed}')
        args_copy = type(args)()  # shallow copy workaround
        for k, v in vars(args).items():
            setattr(args_copy, k, v)
        args_copy.output_dir = seed_output_dir

        # 데이터 준비
        logger.info('Preparing data...')
        dataloaders, num_train_inscope, ood_label_id = prepare_data(args_copy)

        # 모델 로드 (매 시드마다 새로 로드하여 LoRA 초기화)
        model, processor = load_model_and_processor(args_copy)

        # Trainer 생성
        trainer = Qwen3OmniTrainer(
            args=args_copy,
            model=model,
            processor=processor,
            dataloaders=dataloaders,
            num_train_inscope=num_train_inscope,
            ood_label_id=ood_label_id,
        )

        # 학습 전 테스트 파이프라인 검증
        if args_copy.train:
            trainer.preflight_check(n_samples=2)

        # 학습
        if args_copy.train:
            logger.info('Training begins...')
            trainer.train()
            logger.info('Training finished.')

        # 테스트 (학습 안 했으면 저장된 LoRA 가중치 로드 시도, 없으면 zero-shot)
        if not args_copy.train:
            adapter_config = os.path.join(seed_output_dir, 'adapter_config.json')
            if os.path.exists(adapter_config):
                trainer.load_lora_weights(seed_output_dir)
            else:
                logger.info('No saved LoRA weights found — running zero-shot evaluation.')
                trainer.model.disable_adapter_layers()

        logger.info('Testing begins...')
        test_results = trainer.test()
        logger.info('Testing finished.')

        trainer._save_results(test_results)

        # GPU 메모리 정리
        del model, trainer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
