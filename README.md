# intermediate-gradients
Intermediate-gradients is a modification of the Integrated Gradients algorithm.  It is used to obtain the gradients used in approximating the integral of gradients along the path between a baseline and an input.  These gradients can then be used for further attribution exploration.  Intermediate-gradients allows you to look into each of the gradients along this path as well as the respective step size for each gradient.  Additionally, intermediate-gradients supports obtaining gradients from a full layer's inputs or outputs, allowing it to be useful with more complex models.  Based upon the Integrated Gradients implementation from [Captum](https://captum.ai).

## Installation
First, create an Anaconda environment:

		conda create -n intermediate-gradients python=3.8
Then activate the environment and install the requirements:

		conda activate intermediate-gradients
		pip install -r requirements.txt

### IntermediateGradients
A class for calculating the gradients and step sizes.  The number of gradients calculated is directly related to the n_steps parameter
### LayerIntermediateGradients
A class that sets up hooks within the input and output of a full model layer to obtain gradients and step sizes.  Thus, the attributions of more complex models can be evaluated.

## Example scripts
### example.py
This is a quick example script using the [HuggingFace Distilbert](https://huggingface.co/distilbert-base-uncased-distilled-squad) model for question-answering.  It has a customizable steps parameter, so you can see how using more gradients affects the final attributions.  Also, the script has example instructions for turning an intermediate-gradients tensor into an attribution like those calculated by  [Captum's Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients)
After the environment has been set up, this script can be run with 

		python3.8 example.py --n-steps=50