import math
import torch
import torch.nn as nn
import numpy as np
import os
import logging
import pandas as pd

from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datetime import datetime

from config import INTENT_LABELS, OOD_LABEL
from dataset import load_video_frames

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_MULTIMODAL = (
    "You are a multimodal intent recognition system. "
    "Classify the intent of the target utterance into exactly one of these 30 categories:\n"
    "Acknowledge, Advise, Agree, Apologise, Arrange, Ask for help, Asking for opinions, "
    "Care, Comfort, Complain, Confirm, Criticize, Doubt, Emphasize, Explain, "
    "Flaunt, Greet, Inform, Introduce, Invite, Joke, Leave, Oppose, Plan, Praise, "
    "Prevent, Refuse, Taunt, Thank, Warn.\n"
    "Respond with only the intent label, nothing else."
)

SYSTEM_PROMPT_TEXT = (
    "You are an intent recognition system. "
    "Classify the intent of the target utterance into exactly one of these 30 categories:\n"
    "Acknowledge, Advise, Agree, Apologise, Arrange, Ask for help, Asking for opinions, "
    "Care, Comfort, Complain, Confirm, Criticize, Doubt, Emphasize, Explain, "
    "Flaunt, Greet, Inform, Introduce, Invite, Joke, Leave, Oppose, Plan, Praise, "
    "Prevent, Refuse, Taunt, Thank, Warn.\n"
    "Respond with only the intent label, nothing else."
)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, show_results=False):
    """In-scope 분류 메트릭을 계산한다."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    results = {
        'acc': acc,
        'f1': f1,
        'weighted_f1': f1,
        'precision': prec,
        'recall': rec,
    }

    if show_results:
        logger.info(f"  acc={acc:.4f}  f1={f1:.4f}  prec={prec:.4f}  rec={rec:.4f}")

    return results


# ──────────────────────────────────────────────
# LoRA EarlyStopping (어댑터 가중치만 저장)
# ──────────────────────────────────────────────

class LoRAEarlyStopping:

    def __init__(self, patience, monitor='acc', delta=0.0, top_k=3):
        self.patience = patience
        self.monitor = monitor
        self.counter = 0
        self.best_score = float('inf') if monitor == 'loss' else float('-inf')
        self.early_stop = False
        self.delta = delta
        self.best_lora_state = None
        self.top_k = top_k
        self.top_k_checkpoints = []

    def __call__(self, score, model, epoch=None):
        better = (score <= (self.best_score - self.delta)
                  if self.monitor == 'loss'
                  else score >= (self.best_score + self.delta))

        if better:
            self.counter = 0
            self.best_lora_state = self._get_lora_state(model)
            self.best_score = score
        else:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        if epoch is not None:
            self._update_top_k(score, epoch, model)

    def _get_lora_state(self, model):
        return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}

    def _update_top_k(self, score, epoch, model):
        lora_state = self._get_lora_state(model)
        self.top_k_checkpoints.append({'score': score, 'epoch': epoch, 'lora_state': lora_state})
        reverse = (self.monitor != 'loss')
        self.top_k_checkpoints.sort(key=lambda x: x['score'], reverse=reverse)
        if len(self.top_k_checkpoints) > self.top_k:
            self.top_k_checkpoints = self.top_k_checkpoints[:self.top_k]
        logger.info(
            f"Top-{self.top_k}: "
            f"{[(c['epoch'], round(c['score'], 4)) for c in self.top_k_checkpoints]}"
        )

    def get_top_k_checkpoints(self):
        return self.top_k_checkpoints

    def restore(self, model, lora_state=None):
        state = lora_state if lora_state is not None else self.best_lora_state
        if state is None:
            return
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name].to(param.device))


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

class Qwen3OmniTrainer:

    def __init__(self, args, model, processor, dataloaders, num_train_inscope, ood_label_id):

        self.args = args
        self.model = model
        self.processor = processor
        self.dataloaders = dataloaders
        self.ood_label_id = ood_label_id
        self.device = next(model.parameters()).device

        # Label mappings
        self.label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.id2label = {i: label for i, label in enumerate(INTENT_LABELS)}

        # Optimizer (LoRA params only)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

        steps_per_epoch = math.ceil(num_train_inscope / args.gradient_accumulation_steps)
        num_train_steps = steps_per_epoch * args.num_epochs
        num_warmup = max(1, int(num_train_steps * args.warmup_proportion))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup, num_train_steps
        )

        self.best_eval_score = 0
        self.system_prompt = SYSTEM_PROMPT_TEXT if args.modality == 'text' else SYSTEM_PROMPT_MULTIMODAL

    # ──── Prompt construction ────

    def _build_messages(self, utterances, target_idx):
        """Qwen3-Omni 채팅 메시지를 구성한다. context에 OOD 발화도 포함."""
        context_start = max(0, target_idx - self.args.context_len)

        context_lines = []
        for i in range(context_start, target_idx):
            utt = utterances[i]
            context_lines.append(f"[{utt['speaker_name']}]: {utt['text']}")

        target_utt = utterances[target_idx]
        context_str = "\n".join(context_lines) if context_lines else "(No prior context)"

        user_text = (
            f"Conversation context:\n{context_str}\n\n"
            f"Target utterance by [{target_utt['speaker_name']}]: "
            f"\"{target_utt['text']}\"\n\n"
            f"What is the intent of the target utterance?"
        )

        # 비디오 프레임 로드 (텍스트 모드에서는 스킵)
        if self.args.modality == 'text':
            video_frames = None
        else:
            video_frames = load_video_frames(target_utt['video_path'], self.args.num_video_frames)

        content = []
        if video_frames is not None:
            content.append({"type": "video", "video": video_frames})
        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        return messages, video_frames

    # ──── Training ────

    def train(self):
        args = self.args
        early_stopping = LoRAEarlyStopping(
            patience=args.wait_patience,
            monitor=args.eval_monitor,
            top_k=args.top_k_checkpoints,
        )
        accum_steps = args.gradient_accumulation_steps

        for epoch in trange(args.num_epochs, desc="Epoch"):
            self.model.train()
            total_loss = 0.0
            num_samples = 0
            self.optimizer.zero_grad()
            accum_count = 0

            for batch in tqdm(self.dataloaders['train'], desc="Train"):
                dialogue = batch[0]
                utterances = dialogue['utterances']

                for target_idx, utt in enumerate(utterances):
                    if utt['label_id'] == self.ood_label_id:
                        continue

                    messages, video_frames = self._build_messages(utterances, target_idx)
                    target_label_text = self.id2label[utt['label_id']]

                    inputs = self._prepare_training_input(messages, video_frames, target_label_text)
                    if inputs is None:
                        continue

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs)
                    loss = outputs.loss / accum_steps
                    loss.backward()

                    total_loss += outputs.loss.item()
                    num_samples += 1
                    accum_count += 1

                    if accum_count % accum_steps == 0:
                        if args.grad_clip > 0:
                            nn.utils.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                args.grad_clip,
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

            # Flush remaining gradients
            if accum_count % accum_steps != 0:
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            avg_loss = total_loss / max(num_samples, 1)

            # Skip validation before val_start_epoch
            if (epoch + 1) < args.val_start_epoch:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} (validation starts at epoch {args.val_start_epoch})")
                continue

            # Validation
            eval_results = self.evaluate('dev')
            eval_score = eval_results[args.eval_monitor]

            logger.info(
                f"Epoch {epoch+1}: train_loss={avg_loss:.4f}  "
                f"eval_{args.eval_monitor}={eval_score:.4f}  "
                f"best={early_stopping.best_score:.4f}"
            )

            early_stopping(eval_score, self.model, epoch=epoch + 1)

            if early_stopping.early_stop:
                logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        early_stopping.restore(self.model)

        # Save LoRA weights
        if args.save_model:
            self._save_lora_weights(args.output_dir)

        # Test top-k checkpoints
        self._test_top_k(early_stopping)

    def _prepare_training_input(self, messages, video_frames, target_label_text):
        """학습 입력을 준비한다. prompt 토큰에는 -100 마스킹."""
        try:
            # Fix C: 비디오를 포함하여 prompt 길이를 측정 (vision 토큰 확장 반영)
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                videos=[video_frames] if video_frames is not None else None,
                return_tensors="pt",
                padding=False,
            )
            prompt_len = prompt_inputs['input_ids'].shape[1]

            # 전체 텍스트 (assistant 응답 포함)
            messages_with_answer = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": target_label_text}]}
            ]
            full_text = self.processor.apply_chat_template(
                messages_with_answer, tokenize=False, add_generation_prompt=False
            )

            inputs = self.processor(
                text=[full_text],
                videos=[video_frames] if video_frames is not None else None,
                return_tensors="pt",
                padding=True,
            )

            labels = inputs['input_ids'].clone()
            labels[:, :prompt_len] = -100

            # Fix D: padding 토큰도 -100으로 마스킹
            if 'attention_mask' in inputs:
                labels[inputs['attention_mask'] == 0] = -100

            inputs['labels'] = labels

            return inputs
        except Exception as e:
            logger.warning(f'Failed to prepare input: {e}')
            return None

    # ──── Evaluation ────

    def evaluate(self, split='dev', show_results=False):
        self.model.eval()

        all_preds = []
        all_labels = []
        parse_failures = 0

        dataloader = self.dataloaders[split]

        for batch in tqdm(dataloader, desc=f"Eval ({split})"):
            dialogue = batch[0]
            utterances = dialogue['utterances']

            for target_idx, utt in enumerate(utterances):
                if utt['label_id'] == self.ood_label_id:
                    continue

                messages, video_frames = self._build_messages(utterances, target_idx)

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    videos=[video_frames] if video_frames is not None else None,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.args.max_new_tokens,
                        do_sample=False,
                    )

                # padding-safe: attention_mask 합으로 실제 입력 토큰 수 계산
                prompt_len = int(inputs['attention_mask'][0].sum().item())
                new_tokens = generated_ids[0, prompt_len:]
                generated_text = self.processor.decode(
                    new_tokens, skip_special_tokens=True
                )

                pred_id = self._parse_intent(generated_text)
                if pred_id == -1:
                    parse_failures += 1
                    pred_id = 0

                all_preds.append(pred_id)
                all_labels.append(utt['label_id'])

        if parse_failures > 0:
            logger.warning(f'[{split}] Parse failures: {parse_failures}/{len(all_preds)}')

        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)

        results = compute_metrics(y_true, y_pred, show_results=show_results)
        results['y_pred'] = y_pred
        results['y_true'] = y_true

        self.model.train()
        return results

    def _parse_intent(self, generated_text):
        """생성된 텍스트에서 intent label을 파싱한다."""
        text = generated_text.strip()

        if text in self.label2id:
            return self.label2id[text]

        text_lower = text.lower()
        for label, idx in self.label2id.items():
            if label.lower() == text_lower:
                return idx

        for label, idx in self.label2id.items():
            if label.lower() in text_lower:
                return idx

        logger.debug(f'Parse failed: "{text}"')
        return -1

    # ──── Testing ────

    def test(self):
        results = self.evaluate('test', show_results=True)
        if hasattr(self, 'best_eval_score'):
            results['best_eval_score'] = round(self.best_eval_score, 4)
        return results

    def _test_top_k(self, early_stopping):
        top_k = early_stopping.get_top_k_checkpoints()
        if not top_k:
            return

        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Top-{len(top_k)} Checkpoints")
        logger.info(f"{'='*50}")

        current_state = early_stopping._get_lora_state(self.model)

        best_score = -1
        best_results = None
        best_epoch = None

        for rank, ckpt in enumerate(top_k, 1):
            epoch = ckpt['epoch']
            val_score = ckpt['score']

            logger.info(f"\n--- Rank {rank}: Epoch {epoch} (Val: {val_score:.4f}) ---")
            early_stopping.restore(self.model, ckpt['lora_state'])

            test_results = self.evaluate('test', show_results=True)
            test_results['checkpoint_epoch'] = epoch
            test_results['checkpoint_val_score'] = round(val_score, 4)
            test_results['best_eval_score'] = round(val_score, 4)
            test_results['epoch'] = epoch

            self._save_results(test_results)

            score = test_results.get(self.args.eval_monitor, 0)
            if score > best_score:
                best_score = score
                best_results = test_results
                best_epoch = epoch

        early_stopping.restore(self.model, current_state)

        logger.info(f"\n{'='*50}")
        logger.info(f"Best Test: Epoch {best_epoch}, {self.args.eval_monitor}={best_score:.4f}")
        if best_results:
            for k in ['acc', 'f1', 'precision', 'recall']:
                if k in best_results:
                    logger.info(f"  {k}: {best_results[k]:.4f}")
        logger.info(f"{'='*50}\n")

    # ──── Save / Load ────

    def _save_lora_weights(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logger.info(f'LoRA adapter saved to {output_dir}')

    def load_lora_weights(self, model_dir):
        """저장된 LoRA 어댑터를 로드한다.

        PeftModel.from_pretrained로 로드해야 adapter 구조가 정확히 재현된다.
        """
        from peft import PeftModel
        adapter_config = os.path.join(model_dir, 'adapter_config.json')
        if os.path.exists(adapter_config):
            self.model = PeftModel.from_pretrained(self.model, model_dir)
            self.model.to(self.device)
            logger.info(f'LoRA adapter loaded from {model_dir}')
        else:
            logger.warning(f'No adapter_config.json found in {model_dir}')

    def _save_results(self, test_results):
        args = self.args
        results = {}

        for key in ['acc', 'f1', 'weighted_f1', 'precision', 'recall']:
            if key in test_results:
                results[key] = round(test_results[key] * 100, 2)

        if 'best_eval_score' in test_results:
            results[f'eval_{args.eval_monitor}'] = test_results['best_eval_score']
        if 'epoch' in test_results:
            results['epoch'] = test_results['epoch']

        results['model'] = args.model_name
        results['lora_r'] = args.lora_r
        results['context_len'] = args.context_len
        results['seed'] = args.seed
        results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        results_dir = os.path.dirname(os.path.join(args.output_dir, args.results_file))
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(args.output_dir, args.results_file)
        df_new = pd.DataFrame([results])

        if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
            df_old = pd.read_csv(results_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(results_path, index=False)
        logger.info(f'Results saved to {results_path}')
