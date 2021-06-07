"""Lambda to start an A2I human loop to review a non-confident model output

Should be called as an *asynchronous* task from Step Functions (using lambda:invoke.waitForToken)

By passing the Step Functions task token to the A2I task as input, we ensure it gets included in the output
JSON generated by the task and therefore enable our S3-triggered callback function to retrieve the task token
and signal to Step Functions that the review is complete.
"""

# Python Built-Ins:
from datetime import datetime
import json
import os
import re
import uuid

# External Dependencies:
import boto3


a2i = boto3.client("sagemaker-a2i-runtime")
ssm = boto3.client("ssm")

default_flow_definition_arn_param = os.environ.get("DEFAULT_FLOW_DEFINITION_ARN_PARAM")


class MalformedRequest(ValueError):
    pass


def generate_human_loop_name(s3_object_key: str, max_len: int=63) -> str:
    """Create a random-but-a-bit-meaningful unique name for human loop job

    Generated names combine timestamp, object filename, and a random element.
    """
    filename = s3_object_key.rpartition("/")[2]
    filename_component = re.sub(
        # Condense double-hyphens:
        r"--",
        "-",
        re.sub(
            # Cut out any remaining disallowed characters:
            r"[^a-zA-Z0-9\-]",
            "",
            re.sub(
                # Turn significant punctuation to hyphens:
                r"[ _.,!?]",
                "-",
                filename
            )
        )
    )

    datetime_component = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3] # (Millis is enough, no need for microseconds)
    random_component = str(uuid.uuid4()).partition("-")[0] # Most significant bits section of a GUID

    clipped_filename_component = filename_component[:max_len - len(datetime_component) - len(random_component)]

    return f"{datetime_component}-{clipped_filename_component}-{random_component}"[:max_len]


def handler(event, context):
    try:
        execution_context = event["ExecutionContext"]
        task_token = execution_context["Task"]["Token"]

        srcuri = event.get("S3Uri")
        if not srcuri:
            srcbucket = event["Bucket"]
            srckey = event["Key"]
            srcuri = f"s3://{srcbucket}/{srckey}"

        model_result = event["ModelResult"]
        task_input = {
            "taskObject": srcuri,
            # Not used in A2I, purely for feed-through to our callback function:
            "taskToken": task_token,
            # By including our confidence scores in the task input, we open the door for custom task UIs that
            # try to optimize reviewer time by emphasising or de-emphasising particular fields for review:
            "date": {
                "confidence": model_result["Date"]["Confidence"],
                "value": model_result["Date"]["Value"],
            },
            "total": {
                "confidence": model_result["Total"]["Confidence"],
                "value": model_result["Total"]["Value"],
            },
            "vendor": {
                "confidence": model_result["Vendor"]["Confidence"],
                "value": model_result["Vendor"]["Value"],
            },
        }


        if "FlowDefinitionArn" in event:
            flow_definition_arn = event["FlowDefinitionArn"]
        elif default_flow_definition_arn_param:
            flow_definition_arn = ssm.get_parameter(
                Name=default_flow_definition_arn_param,
            )["Parameter"]["Value"]
            if (not flow_definition_arn) or flow_definition_arn.lower() in ("undefined", "null"):
                raise MalformedRequest(
                    "Neither request FlowDefinitionArn nor expected SSM parameter are set. Got: "
                    f"{default_flow_definition_arn_param} = '{flow_definition_arn}'"
                )
        else:
            raise MalformedRequest(
                "FlowDefinitionArn not specified in request and DEFAULT_FLOW_DEFINITION_ARN_PARAM "
                "env var not set"
            )
    except KeyError as ke:
        raise MalformedRequest(f"Missing field {ke}, please check your input payload")

    print(f"Starting A2I human loop with input {task_input}")
    a2i_response = a2i.start_human_loop(
        HumanLoopName=generate_human_loop_name(srcuri),
        FlowDefinitionArn=flow_definition_arn,
        HumanLoopInput={
            "InputContent": json.dumps(task_input)
        },
        DataAttributes={
            "ContentClassifiers": ["FreeOfPersonallyIdentifiableInformation"]
        }
    )
    print(f"Human loop started: {a2i_response}")

    # Doesn't really matter what we return because Step Functions will wait for the callback with the token!
    return a2i_response["HumanLoopArn"]