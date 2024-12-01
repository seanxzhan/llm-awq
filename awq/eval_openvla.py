import os
from collections import deque
import torch
import tqdm
import numpy as np
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from openvla.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from openvla.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
from openvla.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.tensorboard import SummaryWriter

def detokenize_actions(action_token_ids: np.ndarray, vla, action_tokenizer, unnorm_key):
    normalized_actions = action_tokenizer.decode_token_ids_to_actions(action_token_ids)

    # Un-normalize Actions
    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return actions, normalized_actions

def load_model(path):
    # loads in original model
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    vla = AutoModelForVision2Seq.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to('cuda')
    vla.eval()
    return vla


def load_model_lora(path, lora_pt):
    # loads in original model
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    vla = AutoModelForVision2Seq.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # print(vla)
    # exit(0)

    state_dict = torch.load(lora_pt)
    state_dict_good = {}
    for k, v in state_dict.items():
        state_dict_good[k[len("base_model.model."):]] = v

    vla.load_state_dict(state_dict_good)

    vla.to('cuda')
    vla.eval()
    return vla


def evaluate_vla(args, vla_language_backbone: LlamaForCausalLM = None) -> None:
    print(f"Evaluating OpenVLA Model")

    if args.lora_pt is None:
        vla = load_model(args.model_path)
    else:
        vla = load_model_lora(args.model_path, args.lora_pt)
        assert vla_language_backbone is None
    if vla_language_backbone is not None:
        vla.language_model = vla_language_backbone
        print("---------")
        print("yayyy")
        print("---------")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Start =>> Build Directories
    # run_dir = cfg.run_root_dir / exp_id
    run_dir = args.eval_root_dir
    os.makedirs(run_dir, exist_ok=True)
    run_dir = os.path.join(run_dir, args.expname)
    os.makedirs(run_dir, exist_ok=True)

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.model_path else VicunaV15ChatPromptBuilder,
    )
    print("--------------------------")
    print("Training: ", not args.eval_set_test)
    print("--------------------------")
    vla_dataset = RLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=(224, 224),  # 224 is hard coded, originally tuple(vla.module.config.image_sizes)
        shuffle_buffer_size=100_000,
        image_aug=False,
        train = not args.eval_set_test
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=args.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    unnorm_key = "bridge_orig"
    assert vla.get_action_dim("bridge_orig") == 7

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    # recent_losses = deque(maxlen=1)
    # recent_action_accuracies = deque(maxlen=1)
    # recent_l1_losses = deque(maxlen=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(os.path.join(run_dir, "summary"))

    assert args.batch_size == 1

    # Evaluate!
    with tqdm.tqdm(total=len(dataloader), leave=False) as progress:
        for batch_idx, batch in enumerate(dataloader):
            # Always with batch size 1!!!!!!
            assert args.batch_size == 1

            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]

            # DIY WARNING

            # ================================
            # INFERENCE
            # ================================

            # Remove Prompt
            input_ids = input_ids[:, :-8]

            # Add empty token if necessary
            # NB: not necessary becuse I chunked properly here!
            if not torch.all(input_ids[:, -1] == 29871):
                assert False
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )

            # Generate
            generated_ids = vla.generate(
                input_ids=input_ids,  # Shape: [1, seq]
                pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=vla.get_action_dim(unnorm_key),
                attention_mask=attention_mask,
                # Greedy sampling (max-likelihood)
                do_sample=False,
            )

            # ================================
            # DETOKENIZATION
            # ================================

            # Extract predicted action tokens and translate into (normalized) continuous actions
            predicted_action_token_ids = generated_ids[0, -vla.get_action_dim(unnorm_key) :].cpu().numpy()
            predicted_actions, norm_predicted_actions = detokenize_actions(predicted_action_token_ids, vla, action_tokenizer, unnorm_key)

            gt_action_ids = labels[0, -8:-1].cpu().numpy()
            gt_actions, norm_gt_actions = detokenize_actions(gt_action_ids, vla, action_tokenizer, unnorm_key)

            # ================================
            # DETOKENIZATION
            # ================================

            assert len(predicted_action_token_ids) == 7 and len(gt_action_ids) == 7
            action_accuracy = (predicted_action_token_ids == gt_action_ids).sum() / 7.0
            action_l2_loss = np.mean((predicted_actions - gt_actions) ** 2)
            norm_action_l2_loss = np.mean((norm_predicted_actions - norm_gt_actions) ** 2)

            action_accuracy_6 = (predicted_action_token_ids[:6] == gt_action_ids[:6]).sum() / 6.0
            action_l2_loss_6 = np.mean((predicted_actions[:6] - gt_actions[:6]) ** 2)

            # Write to tensorboard
            if distributed_state.is_main_process and batch_idx % 10 == 0:
                writer.add_scalar("action_accuracy", action_accuracy, batch_idx)
                writer.add_scalar("action_l2_loss", action_l2_loss, batch_idx)
                writer.add_scalar("norm_action_l2_loss", norm_action_l2_loss, batch_idx)
                writer.add_scalar("action_accuracy_6", action_accuracy_6, batch_idx)
                writer.add_scalar("action_l2_loss_6", action_l2_loss_6, batch_idx)

            progress.update()

            if batch_idx == len(dataloader):
            # if batch_idx == 100:
                print(f"Dataset length {len(dataloader)} reached! Stopping evaluating...")
                break