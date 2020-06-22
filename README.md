# intermediate-gradients
Intermediate-gradients is a modification of the Integrated Gradients algorithm.  It is used to obtain the gradients used in approximating the integral of gradients along the path between baselines and inputs.  These gradients can then be used for further attribution exploration.  Intermediate-gradients allows you to look into each of the gradients along this path as well as the respective step size for each gradient.  Additionally, intermediate-gradients supports obtaining gradients from a full layer's inputs or outputs, allowing it to be useful with more complex models.  Based upon the Integrated Gradients implementation from [Captum](https://captum.ai).

### IntermediateGradients
A class for calculating the gradients and step sizes.  The number of gradients calculated is directly related to the n_steps parameter
### LayerIntermediateGradients
A class that sets up hooks within the input and output of a full model layer to obtain gradients and step sizes.  Thus, the attributions of more complex models can be evaluated.

## Sample Script using LayerIntermediateGradients
First calculate the gradients:

	# Establish parameters
	>>> N_STEPS = 50
	>>> model.embeddings.num_embeddings
	768
	# Set up input and baseline tensors
	>>> input_ids = torch.tensor([input_ids])
	>>> baseline_ids = torch.tensor([baseline_ids])
	>>> input_ids.shape, baseline_ids.shape
	torch.Size([1, 26]), torch.Size([1, 26])
	# Create an instance of Layer Intermediate Gradients
	>>> lig = LayerIntermediateGradients(forward_func, model.embeddings)
	# Pass the inputs and baselines through the attribute function to obtain the gradients
	>>> grads, start_step_sizes = lig.attribute(inputs=input_ids, baselines=baseline_ids, additional_forward_args=(None, 0), n_steps=N_STEPS)
	>>> intermediate_start.shape, start_step_sizes.shape
	torch.Size([50, 26, 768]), torch.Size([50, 1])

You can then easily calculate the attributions from these gradients and step sizes at a later point.

	# Multiply by the step sizes
	>>> scaled_grads = (intermediate_start.view(N_STEPS, -1) * start_step_sizes)
	# Reshape and sum along the num_steps dimension
	>>> scaled_grads = torch.sum(scaled_grads.reshape((N_STEPS, 1) + intermediate_start.shape[1:]), dim=0)
	# Pass forward the input and baseline ids for reference
	>>> forward_input_ids = model.embeddings.forward(input_ids)
	>>> forward_baseline_ids = model.embeddings.forward(baseline_ids)
	# Multiply the gradients by the difference of the inputs and baselines to get the attributions
	>>> attributions = scaled_grads * (forward_input_ids - forward_baseline_ids)
	>>> attributions.shape
	torch.Size([1, 26, 768])
	