Architecture visualization
==========================

DeepMTP can inspect a validated configuration before constructing the model.
This is useful for checking branch dimensions, composite components, fusion,
output width, and foundation-model training settings without loading data or
downloading model weights.

Install the optional terminal renderer::

  python -m pip install "DeepMTP[visualization]"

Python API
----------

Print a responsive terminal diagram directly from a configuration::

  from DeepMTP import DeepMTPConfig, print_architecture

  config = DeepMTPConfig(
      validation_setting="B",
      problem_mode="classification",
      general_architecture_version="dot_product",
      instance_branch_architecture="MLP",
      instance_branch_input_dim=128,
      instance_branch_nodes_per_layer=[64],
      target_branch_architecture="MLP",
      target_branch_input_dim=32,
      target_branch_nodes_per_layer=[64],
      embedding_size=32,
  )

  description = print_architecture(config)

Rich rendering is selected automatically in an interactive terminal. Output
redirected to a file or CI log uses the dependency-free plain representation.
Use ``style="plain"`` or ``style="rich"`` to make the choice explicit.

``describe_architecture`` returns the renderer-independent semantic model.
Its ``to_dict`` method is suitable for experiment metadata and other tooling::

  from DeepMTP import describe_architecture

  architecture = describe_architecture(config)
  metadata = architecture.to_dict()

Parameter statistics
--------------------

Configuration-only inspection deliberately does not construct a model. Pass
an existing PyTorch module, ``ModelBundle``, or ``DeepMTP`` trainer to add exact
parameter statistics::

  from DeepMTP import ModelFactory, print_architecture

  bundle = ModelFactory(config).build()
  print_architecture(config, model=bundle)

The parameter panel separates trainable and frozen parameters. This makes
foundation-model freezing and LoRA parameter efficiency immediately visible.

Command line
------------

Visualize any saved DeepMTP ``config.json``::

  deepmtp architecture results/my-run/config.json

Select deterministic plain text or machine-readable JSON::

  deepmtp architecture config.json --format plain
  deepmtp architecture config.json --format json

The module entry point is equivalent when the console script is unavailable::

  python -m DeepMTP architecture config.json

SVG export
----------

Rich can save the same diagram as a standalone SVG::

  deepmtp architecture config.json --export-svg architecture.svg

The Python API provides the same operation::

  from DeepMTP import export_architecture_svg

  export_architecture_svg(config, "architecture.svg")

SVG export requires the ``visualization`` extra. It creates parent directories
when necessary and does not instantiate the configured model.
