import os
from collections import deque
import torch
import tqdm
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
    print(vla)
    exit(0)

    state_dict = torch.load(lora_pt)
    vla.load_state_dict(state_dict)

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

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=1)
    recent_action_accuracies = deque(maxlen=1)
    recent_l1_losses = deque(maxlen=1)

    writer = SummaryWriter(os.path.join(run_dir, "summary"))

    # Evaluate!
    with tqdm.tqdm(total=len(dataloader), leave=False) as progress:
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[
                :, 256:-1
            ]  # 256 is hard coded, originally vla.module.vision_backbone.featurizer.patch_embed.num_patches
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            
            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            #print("no_last_bit_eval", args.no_last_bit_eval)
            if args.no_last_bit_eval == True: 
                correct_preds = correct_preds[:6]
                mask = mask[:6]
                #print(f"Shape of correct_preds after slicing: {correct_preds.shape}")
                #print(f"Shape of mask after slicing: {mask.shape}")
                

            action_accuracy = correct_preds.sum().float() / mask.sum().float()


            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            if args.no_last_bit_eval == True: 
                continuous_actions_pred = continuous_actions_pred[:6]
                continuous_actions_gt = continuous_actions_gt[:6]
                #print(f"Shape of continuous_actions_pred after slicing: {continuous_actions_pred.shape}")
                #print(f"Shape of continuous_actions_gt after slicing: {continuous_actions_gt.shape}")

            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and batch_idx % 10 == 0:
                writer.add_scalar("loss", loss, batch_idx)
                writer.add_scalar("action_accuracy", action_accuracy, batch_idx)
                writer.add_scalar("action_l1_loss", action_l1_loss, batch_idx)

            progress.update()

            if batch_idx == len(dataloader):
                print(f"Dataset length {len(dataloader)} reached! Stopping evaluating...")
                break