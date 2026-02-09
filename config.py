import argparse


INTENT_LABELS = [
    'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange',
    'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain',
    'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain',
    'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite',
    'Joke', 'Leave', 'Oppose', 'Plan', 'Praise',
    'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
]

OOD_LABEL = 'UNK'


def parse_args():

    parser = argparse.ArgumentParser(description='Qwen3-Omni MIntRec2.0 In-scope Intent Recognition')

    # Modality
    parser.add_argument('--modality', type=str, default='multimodal',
                        choices=['text', 'multimodal'],
                        help='Input modality: text-only or multimodal (text+video)')

    # Data
    parser.add_argument('--text_data_path', type=str, default='../text_data',
                        help='Path to TSV files (train.tsv, dev.tsv, test.tsv)')
    parser.add_argument('--video_data_path', type=str, default='../video_data',
                        help='Path to video files (dia{id}_utt{id}.mp4)')

    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Omni-30B-A3B-Thinking',
                        help='HuggingFace model ID or local path')

    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', nargs='+',
                        default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    # Training
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--val_start_epoch', type=int, default=2,
                        help='Start validation from this epoch')
    parser.add_argument('--wait_patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--eval_monitor', type=str, default='acc',
                        choices=['acc', 'f1', 'weighted_f1'])
    parser.add_argument('--top_k_checkpoints', type=int, default=3)

    # Video
    parser.add_argument('--num_video_frames', type=int, default=8,
                        help='Number of frames to sample per video')

    # Dialogue
    parser.add_argument('--context_len', type=int, default=1,
                        help='Number of previous utterances as context')

    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=10)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save results and checkpoints')
    parser.add_argument('--results_file', type=str, default='results.csv')
    parser.add_argument('--save_model', action='store_true')

    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='Seeds to iterate over')
    parser.add_argument('--log_dir', type=str, default='logs')

    args = parser.parse_args()
    return args
