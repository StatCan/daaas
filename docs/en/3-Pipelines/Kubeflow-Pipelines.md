# Overview

[Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
is a platform for building machine learning workflows for deployment in a
Kubernetes environment. It enables authoring _pipelines_ that encapsulate
analytical workflows (transforming data, training models, building visuals,
etc.). These _pipelines_ can be shared, reused, and scheduled, and are built to
run on compute provided via Kubernetes.

In the context of the Advanced Analytics Workspace, Kubeflow Pipelines are
interacted with through:

- The [Kubeflow UI](../1-Experiments/Kubeflow.md), where from the Pipelines menu
  you can upload pipelines, view the pipelines you have and their results, etc.
- The Kubeflow Pipelines python
  [SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/), accessible
  through the
  [Jupyter Notebook Servers](../1-Experiments/Kubeflow.md#create-a-server),
  where you can define your components and pipelines, submit them to run now, or
  even save them for later.

<!-- prettier-ignore -->
??? example "More examples in the notebooks"
    More comprehensive pipeline examples specifically made for this platform are
    available on [GitHub](https://github.com/StatCan/jupyter-notebooks) (and in
    every Notebook Server at `/jupyter-notebooks`). You can also check out
    [public sources](https://github.com/kubeflow/pipelines/tree/master/samples).

See
[the official Kubeflow docs](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
for a more detailed explanation of Kubeflow Pipelines.

![A Kubeflow Pipeline](../images/kf-pipeline_with_result.png)

# What are Pipelines and How do they Work?

A
[_pipeline_](https://www.kubeflow.org/docs/pipelines/overview/concepts/pipeline/)
in Kubeflow Pipelines consists of one or more
[_pipeline components_](https://www.kubeflow.org/docs/pipelines/overview/concepts/component/)
chained together to form a workflow. The components are like functions, and then
the pipeline just connects them together.

The _pipeline_ describes the entire workflow of what you want to accomplish,
while the _pipeline components_ each describe individual steps in that process
(such as pulling columns from a data store, transforming data, or training a
model). Each _component_ should be **modular**, and ideally **reusable**.

At their core, each _component_ has:

- A standalone application, packaged as a
  [docker images](https://docs.docker.com/get-started/), for doing the actual
  work. The code in the docker image could be a shell script, python script, or
  anything else you can run from a Linux terminal
- A description of how Kubeflow Pipelines runs the code (where is the image,
  what command line arguments does it accept, what outputs does it generate), as
  a YAML file

A _pipeline_ then, using the above _components_, defines the logic for how
_components_ are connected, such as:

1. run ComponentA
2. pass the output from ComponentA to ComponentB and ComponentC
3. ...

<!-- prettier-ignore -->
!!! example "Example of a pipeline" Here's an example

        #!/bin/python3
        dsl.pipeline(
            name="Estimate Pi",
            description='Estimate Pi using a Map-Reduce pattern'
        )
        def compute_pi():

            # Create a "sample" operation for each seed passed to the pipeline
            seeds = (1,2,3,4,5,6,7,8,9,10)
            sample_ops = [sample_op(seed) for seed in seeds]

            # Get the results, before we feed into two different pipelines.
            # The results are extracted from the output_file.json files,
            # are available from the sample_op instances through the .outputs attribute
            outputs = [s.outputs['output'] for s in sample_ops]

            _generate_plot_op = generate_plot_op(outputs)
            _average_op = average_op(outputs)

    You can find the full pipeline in the [`map-reduce-pipeline` example](https://github.com/StatCan/jupyter-notebooks)

# Define and run your first Pipeline

While _pipelines_ and _components_ are defined by YAML files, the python
[SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/) let's you
define them from python code. The following is an example of how to define a
[simple pipeline](https://github.com/StatCan/jupyter-notebooks/blob/master/kfp-basics/average_with_docker_components.ipynb)
using the python SDK.

The objective of our pipeline is, given five numbers, compute:

1. The average of the first three numbers
2. The average of the last two numbers
3. The average of the results of (1) and (2)

To do this, we define a _pipeline_ that uses our average _component_ to do the
computations.

The _average_ component is defined by a docker image with a simple python script
that:

- accepts one or more numbers as command line arguments
- returns the average of these numbers, written to the file `out.txt` within its
  container

To tell Kubeflow Pipelines how to use this image, we define our average
_component_ through a ContainerOp which tells Kubeflow our image's API. The
ContainerOp instance sets the docker image location, how to pass arguments to
it, and what outputs to pull from the container. To actually use these
ContainerOp's in our pipeline, we build factory functions like `average_op` (as
we'll probably want more than just one average _component_).

```python
from kfp import dsl

def average_op(*numbers):
    """
    Factory for average ContainerOps

    Accepts an arbitrary number of input numbers, returning a ContainerOp that
    passes those numbers to the underlying docker image for averaging

    Returns output collected from ./out.txt from inside the container

    """
    # Input validation
    if len(numbers) < 1:
        raise ValueError("Must specify at least one number to take the average of")

    return dsl.ContainerOp(
        name="averge",  # What will show up on the pipeline viewer
        image="k8scc01covidacr.azurecr.io/kfp-components/average:v1",  # The image that KFP runs to do the work
        arguments=numbers,  # Passes each number as a separate (string) command line argument
        # Script inside container writes the result (as a string) to out.txt, which
        # KFP reads for us and brings back here as a string
        file_outputs={'data': './out.txt'},
    )
```

We define our pipeline as a python function that uses our above ComponentOp
factories, decorated by the @dsl.pipeline decorator. Our pipeline uses our
_average_ component by passing it numbers, and we use the _average_ results by
passing them to later functions through accessing `avg_*.output`.

```python
@dsl.pipeline(
    name="my pipeline's name"
)
def my_pipeline(a, b, c, d, e):
    """
    Averaging pipeline which accepts five numbers and does some averaging
    operations on them
    """
    # Compute averages for two groups
    avg_1 = average_op(a, b, c)
    avg_2 = average_op(d, e)

    # Use the results from _1 and _2 to compute an overall average
    average_result_overall = average_op(avg_1.output, avg_2.output)
```

And finally, we save a YAML definition of our pipeline for later passing to
Kubeflow Pipelines. This YAML describes to Kubeflow Pipelines exactly how to run
our pipeline - unzip it and take a look yourself!

```python
from kfp import compiler
pipeline_yaml = 'pipeline.yaml.zip'
compiler.Compiler().compile(
    my_pipeline,
    pipeline_yaml
)
print(f"Exported pipeline definition to {pipeline_yaml}")
```

<!-- prettier-ignore -->
??? warning "Kubeflow Pipelines is a lazy beast"
    It is useful to keep in mind what computation is happening when you run this
    python code versus what happens when you submit the pipeline to Kubeflow
    Pipelines. Although it seems like everything is happening in the moment, try
    adding `print(avg_1.output)` to the above pipeline and see what happens when
    you compile your pipeline. The python SDK we're using is for _authoring_
    pipelines, not for running them, so results from components will never be
    available when you run this python code. The is discussed more below in
    _Understanding what computation occurs when_.

To actually run our pipeline, we define an experiment:

```python
experiment_name = "averaging-pipeline"

import kfp
client = kfp.Client()
exp = client.create_experiment(name=experiment_name)

pl_params = {
    'a': 5,
    'b': 5,
    'c': 8,
    'd': 10,
    'e': 18,
}

```

Which is observable in the
[Kubeflow Pipelines UI](../1-Experiments/Kubeflow.md):

![Kubeflow Pipeline Experiment](../images/kfp_experiment.png)

And then run an instance of our pipeline with the arguments we want:

```python
import time

run = client.run_pipeline(
    exp.id,  # Run inside the above experiment
    experiment_name + '-' + time.strftime("%Y%m%d-%H%M%S"),  # Give our job a name with a timestamp so its unique
    pipeline_yaml,  # Pass the .yaml.zip we created above.  This defines the pipeline
    params=pl_params  # Pass our parameters we want to run the pipeline with
)
```

which can also be seen in the UI:

![Kubeflow Pipeline Experiment](../images/kfp_experiment.png)

Later when we want to reuse the pipeline, we can pass different arguments and do
it all again (or even reuse it from within the Kubeflow UI).

To understand this example further, open it up in Kubeflow and try it for
yourself.

# Lightweight components

Under construction, sorry!

# Understanding what computation occurs when

Under construction, sorry!
