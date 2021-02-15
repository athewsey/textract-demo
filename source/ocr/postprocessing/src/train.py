"""Train HuggingFace LayoutLM on Textract results"""

# Python Built-Ins:
from collections import defaultdict
import json
import logging
import os
import pickle
import random
import shutil
import subprocess
import sys
import zipfile

# External Dependencies:
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, LayoutLMForTokenClassification, LayoutLMTokenizer

# Local Dependencies:
import config
import data
# Import everything defined by inference.py to enable directly deploying this model via SageMaker SDK's
# Estimator.deploy(), which will leave env var SAGEMAKER_PROGRAM=train.py:
from inference import *

# TODO: Checkpointing and use-best-params
# TODO: More optimizer params for HPO tuning
# TODO: Distributed

logger = logging.getLogger()


def set_seed(seed, use_gpus=True):
    """Seed all the random number generators we can think of for reproducibility"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_gpus:
            torch.cuda.manual_seed_all(seed)


# TODO: args.num_labels
def get_model(args, device):
    model = LayoutLMForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=args.num_labels
    )
    model.to(device)
    return model


def evaluate(model, dataloader, device, pad_token_label_id):
    loss = 0
    n_examples = 0
    acc_pct_cumsum = 0
    focus_acc_cumsum = 0
    label_pred_counts = defaultdict(int)
    with torch.no_grad():
        for batch in dataloader:
            for name in batch:
                batch[name] = batch[name].to(device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            outputs = model(**batch)  # input_ids, bbox, attention_mask, token_type_ids, labels
            batch_loss_np = (outputs.loss.cpu() if outputs.loss.is_cuda else outputs.loss).numpy()
            loss += batch_loss_np
            n_examples += input_ids.shape[0]
            # outputs.logits is (batch_size, sequence_length, config.num_labels)
            # labels / pred_labels is (batch_size, sequence_length)
            pred_labels = torch.argmax(outputs.logits, dim=2)
            pred_labels_cpu = (pred_labels.cpu() if pred_labels.is_cuda else pred_labels)

            # Update predictions-by-label tracker:
            unique_labels, unique_counts = torch.unique(pred_labels, return_counts=True)
            unique_counts_numpy = (unique_counts.cpu() if unique_counts.is_cuda else unique_counts).numpy()
            unique_labels_numpy = (unique_labels.cpu() if unique_labels.is_cuda else unique_labels).numpy()
            for ix, label in enumerate(unique_labels_numpy):
                label_pred_counts[label] += unique_counts_numpy[ix]

            # Accuracy ignoring PAD, CLS and SEP tokens:
            non_pad_labels = (labels != pad_token_label_id)
            n_tokens_by_example = non_pad_labels.sum(dim=1)
            n_correct_by_example = torch.logical_and(labels == pred_labels, non_pad_labels).sum(dim=1)
            acc_by_example = torch.true_divide(n_correct_by_example, n_tokens_by_example)
            acc_pct_cumsum += torch.sum(acc_by_example)

            # Accuracy ignoring PAD, CLS, SEP tokens *and* tokens where both pred and actual classes are
            # 'other':
            other_class_label = args.num_labels - 1
            focus_labels = torch.logical_and(
                non_pad_labels,
                torch.logical_or(labels != other_class_label, pred_labels != other_class_label),
            )
            n_focus_tokens_by_example = focus_labels.sum(dim=1)
            n_correct_by_example = torch.logical_and(labels == pred_labels, focus_labels).sum(dim=1)
            focus_acc_cumsum += torch.sum(
                torch.true_divide(n_correct_by_example, n_focus_tokens_by_example)
            )
    by_label_preds_total = sum([v for k, v in label_pred_counts.items()])
    logger.info("Evaluation ratios: {}".format({
        k: v / by_label_preds_total
        for k, v in label_pred_counts.items()
    }))
    return {
        "n_examples": n_examples,
        "loss_avg": loss / n_examples,
        "acc": acc_pct_cumsum / n_examples,
        "focus_acc": focus_acc_cumsum / n_examples,
    }


def train(args):
    logger.info("Creating config and model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)

    # TODO: Replace data loader logic
    logger.info("Loading datasets")
    # TODO: Support training custom tokenizer
    tokenizer = LayoutLMTokenizer.from_pretrained(args.base_model)
    # TODO: Should pass this pad ID through to the dataset?
    pad_token_label_id = CrossEntropyLoss().ignore_index
    train_dataset = data.get_dataset(args.train, tokenizer, args)
    logger.info(f"train dataset has {len(train_dataset)} samples")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
    )
    if args.validation:
        val_dataset = data.get_dataset(args.validation, tokenizer, args)
        logger.info(f"validation dataset has {len(val_dataset)} samples")
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.batch_size,
        )
    else:
        val_dataset = None
        val_dataloader = None

    logger.info("Preparing optimizer")
    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0

    # Put the model in training mode:
    model.train()
    logger.info("Training...")
    for epoch in range(args.max_epochs):
        epoch_train_loss = 0
        epoch_train_samples = 0
        for batch in train_dataloader:
            epoch_train_samples += len(batch["input_ids"])
            for name in batch:
                batch[name] = batch[name].to(device)
            outputs = model(
                **batch,  # input_ids, bbox, attention_mask, token_type_ids, labels
            )
            loss = outputs.loss
            epoch_train_loss += loss

            # Print loss every 100 steps:
            if global_step % 100 == 0:
                logger.info(f"Global Step {global_step} Metrics: train.loss={loss.item()};")

            # Backward pass to get the gradients:
            loss.backward()

            # TODO: Consider printing diagnostic gradients on cls head too?
            #print(model.classifier.weight.grad[6,:].sum())

            # Update:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Epoch metrics:
        epoch_metrics = {
            "train.n_examples": epoch_train_samples,
            "train.loss_avg": epoch_train_loss / epoch_train_samples,
        }
        if args.validation:
            logger.info(f"Evaluating epoch {epoch}")
            model.eval()
            val_metrics = evaluate(model, val_dataloader, device, pad_token_label_id)
            model.train()
            for k, v in val_metrics.items():
                epoch_metrics[f"validation.{k}"] = v
        logger.info("Epoch {} Metrics: {};".format(
            epoch,
            "; ".join([
                f"{name}={val}" for name, val in epoch_metrics.items()
            ])
        ))

    logger.info(f"Saving model to {args.model_dir}")
    # The below underlying limitation in TorchServe (which the SM PyTorch v1.6 container uses for serving
    # stack) means that only one 'model.pth' file is copied (and required) from the source directory to the
    # serving area by the inference server.
    # https://github.com/pytorch/serve/pull/814
    #
    # Since our model wants to save multiple artifacts, we'll construct a "model.pth" file which is actually
    # a zip archive of all the files we really want, then provide a custom `model_fn()` at inference time to
    # unpack and load this archive correctly, rather than trying to load it as a raw PyTorch model file.
    #
    # For a more conventional PyTorch model saving example, see:
    # https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/6936c08581e26ff3bac26824b1e4946ec68ffc85/src/sagemaker_pytorch_serving_container/torchserve.py#L45
    model_tmp_dir = os.path.join(args.model_dir, "tmp")
    model.save_pretrained(model_tmp_dir)
    tokenizer_save_dir = os.path.join(model_tmp_dir, "tokenizer")
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_dir)
    with zipfile.ZipFile(
        os.path.join(args.model_dir, "model.pth"),
        "w",
        # Default ZIP_STORED does not compress, but our overall model.tar.gz artifact will be compressed
        # anyway so ehh
        #compression=zipfile.ZIP_DEFLATED,
    ) as mdlzip:
        for cur_path, dirs, files in os.walk(model_tmp_dir):
            for file in files:
                file_path = os.path.join(cur_path, file)
                mdlzip.write(
                    file_path,
                    arcname=file_path[len(model_tmp_dir):]
                )
    shutil.rmtree(model_tmp_dir)

    # To enable directly deploying this model via SageMaker SDK's Estimator.deploy() (rather than needing to
    # create a PyTorchModel with entry_point / source_dir args), we need to save any inference handler
    # function code to model_dir/code. Here we compromise efficiency to the benefit of usage simplicity, by
    # just copying the contents of this training code folder to the model/code folder for inference:
    code_path = os.path.join(args.model_dir, "code")
    logger.info(f"Copying code to {code_path} for inference")
    for currpath, dirs, files in os.walk("."):
        for file in files:
            # Skip any filenames starting with dot:
            if file.startswith("."):
                continue
            filepath = os.path.join(currpath, file)
            # Skip any pycache or dot folders:
            if ((os.path.sep + ".") in filepath) or ("__pycache__" in filepath):
                continue
            relpath = filepath[len("."):]
            if relpath.startswith(os.path.sep):
                relpath = relpath[1:]
            outpath = os.path.join(code_path, relpath)
            logger.info(f"Copying {filepath} to {outpath}")
            os.makedirs(outpath.rpartition(os.path.sep)[0], exist_ok=True)
            shutil.copy2(filepath, outpath)
    return model

# Since we want to support targeting this train.py as a valid import entry point for inference too, we need
# to only run the actual training routine if run as a script, not when imported as a module:
if __name__ == "__main__":
    try:
        with open("/usr/local/cuda/version.txt", "r") as f:
            print("CUDA version:")
            print(f.read())
    except:
        print("Unable to determine CUDA version")

    args = config.parse_args()

    for l in (logger, data.logger):
        config.configure_logger(l, args)

    logger.info("Loaded arguments: %s", args)
    logger.info("Starting!")
    set_seed(args.seed, use_gpus=args.num_gpus > 0)

    # Start training:
    train(args)
