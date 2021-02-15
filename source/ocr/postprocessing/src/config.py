"""SageMaker configuration parsing for Textract LayoutLM
"""

# Python Built-Ins:
import argparse
import json
import logging
import os
import sys

# External Dependencies:
import torch


def configure_logger(logger, args):
    """Configure a logger's level and handler (since base container already configures top level logging)"""
    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(consolehandler)
    logger.setLevel(args.log_level)


def boolean_hyperparam(raw):
    """Boolean argparse type for convenience in SageMaker
    SageMaker HPO supports categorical variables, but doesn't have a specific type for booleans -
    so passing `command --flag` to our container is tricky but `command --arg true` is easy.
    Using argparse with the built-in `type=bool`, the only way to set false would be to pass an
    explicit empty string like: `command --arg ""`... which looks super weird and isn't intuitive.
    Using argparse with `type=boolean_hyperparam` instead, the CLI will support all the various
    ways to indicate 'yes' and 'no' that you might expect: e.g. `command --arg false`.
    """
    valid_false = ("0", "false", "n", "no", "")
    valid_true = ("1", "true", "y", "yes")
    raw_lower = raw.lower()
    if raw_lower in valid_false:
        return False
    elif raw_lower in valid_true:
        return True
    else:
        raise argparse.ArgumentTypeError(
        f"'{raw}' value for case-insensitive boolean hyperparam is not in valid falsy "
        f"{valid_false} or truthy {valid_true} value list"
    )


def list_hyperparam(raw):
    """Basic comma-separated list argparse type for convenience in SageMaker
    No escaping of commas supported, no conversion from raw string type (see list_hyperparam_withparser).
    """
    return [] if raw is None else raw.split(",")


def list_hyperparam_withparser(parser, default=None, ignore_error=False):
    # Define separate functions, rather than using logic, as it's easy & Python try/except can be expensive
    def unsafe_mapper(val):
        result = parser(val)
        return default if result is None else result

    def safe_mapper(val):
        try:
            result = parser(val)
            return default if result is None else result
        except Exception:
            return default

    return lambda raw: list(map(safe_mapper if ignore_error else unsafe_mapper, list_hyperparam(raw)))


def parse_args(cmd_args=None):
    """Parse config arguments from the command line, or cmd_args instead if supplied"""
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    parser = argparse.ArgumentParser(description="Train LayoutLM on Textract data")

    ## Network parameters:
    parser.add_argument("--base-model", type=str,
        default=hps.get("annotation-atr", "microsoft/layoutlm-base-uncased"),
        help="ID of the base model to use from Transformers",
    )
    parser.add_argument("--max-seq-len", type=int, default=hps.get("max-seq-len", 512),
        help="Maximum sequence length this model can be used with",
    )
    parser.add_argument("--num-labels", type=int, default=hps.get("num-labels", 2),
        help="Number of classes",
    )

    ## Training process parameters:
#     parser.add_argument(
#         "--epsilon", type=float, default=hps.get("epsilon", 1e-15),
#         help="If you need help, you don't need to change me.",
#     )
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)"
    )
#     parser.add_argument(
#         "--momentum", type=float, default=hps.get("momentum", 0.02),
#         help="Momentum for batch normalization - typical range 0.01-0.4."
#     )
    parser.add_argument("--lr", "--learning-rate", type=float,
        default=hps.get("lr", hps.get("learning-rate", 5e-5)),
        help="Learning rate (main training cycle)"
    )
#     parser.add_argument(
#         "--clip-value", type=float, default=hps.get("clip-value"),
#         help="Optional gradient value clipping"
#     )
    # device_name param not necessary
    parser.add_argument("--max-epochs", type=int, default=hps.get("max-epochs", 5),
        help="Maximum number of epochs for training"
    )
#     parser.add_argument("--patience", type=int, default=hps.get("patience", 15),
#         help="Number of consecutive epochs without improvement before early stopping"
#     )
    parser.add_argument("--batch-size", type=int, default=hps.get("batch-size", 2),
        help="Number of examples per batch. FUNSD fine-tuning example uses 2."
    )

    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 0),
        help="Number of GPUs to use in training."
    )
    parser.add_argument("--num-workers", "-j", type=int,
        default=hps.get("num-workers", max(0, int(os.environ.get("SM_NUM_CPUS", 0)) - 2)),
        help="Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful"
    )

    # I/O Settings:
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )
    parser.add_argument("--annotation-attr", type=str, default=hps.get("annotation-attr", None),
        help="Attribute name of the annotations in the manifest file"
    )
    parser.add_argument("--textract-prefix", type=str, default=hps.get("textract-prefix", ""),
        help="Prefix mapping manifest S3 URIs to the 'textract' channel"
    )
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--output-data-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--textract", type=str, default=os.environ.get("SM_CHANNEL_TEXTRACT"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args(args=cmd_args)

    ## Post-argparse validations & transformations:
    if not (args.train and args.textract):
        parser.error(
            "'train' (JSONLines manifest file) and 'textract' (Folder of Textract result JSONs) channels "
            "are mandatory"
        )

    # Set up log level: (Convert e.g. "20" to 20 but leave "DEBUG" alone)
    try:
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    # Note basicConfig has already been called by our parent container, so calling it won't do anything.
    logger = logging.getLogger("config")
    configure_logger(logger, args)

    if args.num_gpus and not torch.cuda.is_available():
        parser.error(
            f"Got --num-gpus {args.num_gpus} but torch says cuda is not available: Cannot use GPUs"
        )

    return args
