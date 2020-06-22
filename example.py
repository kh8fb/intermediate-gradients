"""
A quick example script using Distilbert to show intermediate-gradient's capabilities and usage.
"""

from captum.attr import LayerIntegratedGradients
from intermediate_gradients.layer_intermediate_gradients import LayerIntermediateGradients
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertConfig
import torch


def predict(model, inputs):
    """
    Get the model's prediction based upon the combined question/text input tensor.

    Parameters
    ----------
    model: transformers.modeling_distilbert.Distilbert
        Model to run predictions on.
    inputs: torch.tensor(1, num_ids), dtype=int64
        Tensor of the combined tokenized ids from the question and text.
    Returns
    -------
    prediction: (torch.tensor, torch.tensor) each (1, num_ids), dtype=float32
        First element is the model's predicted starting position scores.
        Second element is the model's predicted ending position scores.
    """
    prediction = model(inputs, attention_mask=attention_mask,)
    return prediction


def squad_pos_forward_func(model, inputs, position=0):
    """
    Get the value of the largest starting/ending position from the prediction.

    Parameters
    ----------
    model: transformers.modeling_distilbert.Distilbert
        Model to run predictions on.
    inputs: torch.tensor(1, num_ids), dtype=int64
        Tensor of the combined tokenized ids from the question and text.
    position: int
        Determine whether starting or ending positions are selected.
        If 0, select the highest starting position value.
        If 1, select the highest ending position value.
    Returns
    -------
    max_pred: torch.tensor(1), dtype=float32
        If position is 0, this is the highest value for a starting position prediction
        If position is 1, this is the highest value for an ending position prediction.
    """
    pred = predict(model, inputs)
    pred = pred[position]
    max_pred = pred.max(1).values
    return max_pred


@click.command(help="""Run intermediate gradients on a predetermined input and baseline tensor.
This example script allows you to see the shapes and behavior of the returned gradients, as well as
how to turn them into the attributions from integrated gradients.""")
@click.option(
    "--n-steps",
    help="The number of steps used by the approximation method. Default is 50.",
    required=False,
    default=50,
)
def main(n_steps=50):
    N_STEPS = int(n_steps)

    # load the model and tokenizer
    model_path = 'distilbert-base-uncased-distilled-squad'
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    model.zero_grad()
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    # tokenize the question
    question = "what is important to us?"
    text = "it is important to us to include, empower, and support humans of all kinds"
    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
    input_ids = torch.tensor([tokenizer.encode(input_text)])

    # create a baseline of zeros in the same shape as the inputs
    baseline_ids = torch.zeros_like(input_ids)

    # create an instance of layer intermediate gradients based upon the embedding layer
    lig = LayerIntermediateGradients(squad_pos_forward_func, model.distilbert.embeddings)
    start_grads, start_step_sizes = lig.attribute(inputs=input_ids,
                                                  baselines=baseline_ids,
                                                  additional_forward_args=(0))
    print("Shape of the returned gradients: ")
    print(start_grads.shape)
    print("Shape of the step sizes: ")
    print(start_step_sizes.shape)

    # now calculate attributions from the intermediate gradients
    
    # multiply by the step sizes
    scaled_grads = start_grads.view(N_STEPS, -1) * start_step_sizes
    # reshape and sum along the num_steps dimension
    scaled_grads = torch.sum(scaled_grads.reshape((N_STEPS, 1) + start_grads.shape[1:]), dim=0)
    # pass forward the input and baseline ids for reference
    forward_input_ids = model.distilbert.embeddings.forward(input_ids)
    forward_baseline_ids = model.embeddings.forward(baseline_ids)
    # multiply the scaled gradients by the difference of inputs and baselines to obtain attributions
    attributions = scaled_grads * (forward_input_ids - forward_baseline_ids)
    print("Attributions calculated from intermediate gradients: ")
    print(attributions.shape)
    print(attributions)

    # compare to layer integrated gradients
    layer_integrated = LayerIntegratedGradients(squad_pos_forward_func, model.distilbert.embeddings)
    attrs_start = layer_integrated.attribute(inputs=input_ids,
                                             baselines=baseline_ids,
                                             additional_forward_args=(0),
                                             return_convergence_delta=False)
    print("Attributions from layer integrated gradients: ")
    print(attrs_start.shape)
    print(attrs_start)
    

if __name__=="__main__":
    #pylint: disable=no-parameters
    main()
