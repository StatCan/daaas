**Note: This is an advanced topic and this guide is under construction. Running
this type of job is not trivial and requires Kubernetes, Docker, and PyTorch
experience. Before getting too deep, read this document and decide if it makes
sense for you**

**This only handles PyTorch distributed training, although Tensorflow has very
similar functionality using the TFJob operator**

# Summary

For large training tasks, [PyTorch](https://pytorch.org/) allows for
[distributed training](https://pytorch.org/docs/stable/distributed.html) of
models across multiple GPUs. For the Advanced Analytics Workspace (AAW),
Distributed Data Parallel (DDP) is the best method.

<!-- prettier-ignore -->
!!! note "PyTorch has several distributed methods - make sure you're reading about Distributed Data Parallel"
    If reading tutorials or blogs on your own, be careful to follow instructions for **Distributed** Data Parallel (DDP) and not Data Parallel.  Data Parallel is only suitable for multiple GPUs on a single machine (and even then, might be worse than DDP in many cases).  The nomenclature is very similar (`.to_parallel()` might create a DP model, whereas a `.to_distributed()` creates a DDP model) so be on your toes!

DDP allows for training a single model across multiple GPUs that are on or more
physical machines. DDP replicates the model on every GPU, with each GPU fed a
separate set of inputs (e.g.: images, text, etc., whatever you're training on).
DDP communicates the gradients between all workers to keep them in sync. When
doing DDP, you're effectively scaling up the number of images you process on
each batch during training. A good summary of DDP is given
[here](https://pytorch.org/tutorials/beginner/dist_overview.html).

The following describes how to use DDP jobs through Kubeflow and the AAW. Also
described below are some tips for distributing older fast.ai/PyTorch jobs, which
in general works the same but requires additional effort due to older APIs.
Additional resources on DDP jobs in PyTorch can be found in
[this tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

# Do I need Distributed Training?

Distributed training is a helpful tool, but not helpful for all problems.
Spreading training across multiple GPUs can be beneficial, but also adds
overhead (coordinating training between all the GPUs, handling data, etc.). As a
general guide:

- if you have long, slow training runs (in the scale of hours or more), then DDP
  may be beneficial
- if you have quick training runs (minutes or less) or your epochs are fast (a
  few minutes or less), your problem likely won't be helped by distributed
  training

# Running a "distributed" Hello World Locally (on one or more GPU)

DDP jobs require some small setup and code modifications. Below describes a very
simple DDP job where we split training into two processes (master and worker)
and run them on a single machine. This lets us debug our code and confirm
workers are synchronizing properly before going to multiple machines.

<!-- prettier-ignore -->
??? warn "At time of writing, the AAW only has machines with a single GPU, so this configuration will not result in a speedup of training" 
    If we had more GPUs this would be a perfectly viable way of splitting training across them, but since we only have one we assign both master and worker to the **same GPU**, negating any benefit.

## Code changes required for Distributed Data Parallel training

Relative to a regular single-GPU training run, only a few code changes are
required:

- at some point before training, you must call
  `torch.distributed.init_process_group()`. This code connects the current
  worker to others on the job
- for your `Learner`, you must invoke `learner.to_distributed()` prior to
  training. Depending on the version of PyTorch or fast.ai you are using, you
  may need to use a different `Learner` that has this method (or pull the
  relevant code from the current PyTorch/fast.ai github repositories to extend
  your current version's `Learner`)

<!-- prettier-ignore -->
??? danger "If you forget to call `learner.to_distributed()`, it will result in a subtle bug"
    If you `init_process_group()` but forget to call `.to_distributed()` on your learners, **it will appear** that everything is training normally.  All nodes will show progress and GPU utilization, except that they will all be training **separate models** rather than together training a single model.  You can spot this by looking at the worker logs (worker logs should not report individual training iteration losses, just losses at each epoch), and because the number of iterations required per epoch will be `n_images / batch_size` when it should actually be `n_images / (batch_size * n_gpu)`.  This is a difficult thing to notice, so be very careful always invoke `.to_distributed()`

## Example using MNIST

_Under construction_

The code from
[this example](https://github.com/kubeflow/pytorch-operator/tree/master/examples/mnist)
is a simple example that can be modified for local running. Change
[this line](https://github.com/kubeflow/pytorch-operator/blob/282cbee0f43d2510bf4ac143fbfa86d1d323e2d5/examples/mnist/mnist.py#L116)
to:

```
dist.init_process_group(
	rank=rank,
	world_size=world_size,
	init_method='file:///tmp/torchsharedfile',
)
```

Then run two instances of the same `mnist.py` (in different terminals), setting
:

- `world_size=2`
- (in the first terminal) `rank=0`
- (in the second terminal) `rank=1` (you can do this many ways, e.g.: by saving
  two versions of the code, using environment variables, or through adding
  `rank` and `world_size` command line arguments)

When you run the first, you'll see the script pauses and waits before training.
When the second script is run and catches up, they'll both start training
together (with more output shown in the terminal of the `rank=0` case).

## Tuning Training Parameters of a Distributed Job

Typically this process is similar to any other training job, but see Effective
Batch Size below for more specifics.

# Launching a Distributed Data Parallel job using the PyTorchJob Operator

To assist with training PyTorch models, Kubeflow has a PyTorch Job operator that
handles some of the plumbing behind the scenes. This operator automatically sets
the environment so your distributed jobs synchronize across machines, reducing
the effort from the user. To use this operator, one must:

- (usually slightly) modify your code to take advantage of DDP, similar to shown
  above
- package your code and training data into a Kubernetes/PyTorch Job friendly
  format

This is described more below.

## Code changes required for Distributed Data Parallel training

Code changes required are very similar to those shown above for a local DDP. The
only difference is that, you call:

```
torch.distributed.init_process_group()
```

instead of

```
torch.distributed.init_process_group(
	rank=SOME_RANK,
	world_size=WORLD_SIZE,
	init_method=SOME_INIT_METHOD,
)
```

PyTorch can infer these values because the PyTorch Job operator has set the
environment up for you.

## Packaging Code for a PyTorch Job

Like other Kubernetes Containers, PyTorch Jobs are created from Docker images.
Defining these images is beyond the scope here, but they should have everything
required for your code to run, such as:

- any libraries/binaries needed (e.g.: GPU drivers)
- python and required packages installed (or, if anything has to be installed at
  runtime, a script that will do that for you)
- an entrypoint (could be your python code (some shell script that runs
  everything you need, etc.)

Kubernetes will eventually run this container by executing a single command line
call on the container (just like how you can `docker run` something) using
arguments you can specify below when submitting the PyTorch job.

<!-- prettier-ignore -->
??? danger "Not all Docker images are allowed on AAW"
    For security reasons, the AAW restricts which Docker images can be run.  Some common public Dockerhub repositories, such as `python` and `tensorflow`, are whitelisted, but must (including your own personal Dockerhub repository) are blocked.  User images must be submitted/built using the [daaas-containers](https://github.com/StatCan/daaas-containers) repository by pull request and vetted by the AAW team."

## Accessing Training Data during Training

Every worker needs access to your training data. Several options exist, but all
have complications that make them challenging at present

### Persistent Volume Claims

You can use a Persistent Volume Claim (PVC) (e.g.: the Data Volumes used by
Notebook Servers) to pass training data to your process by pre-loading one (or
more, see note below). You load the PVC with your training data, then you
provide the PVC to your master/worker at PyTorchJob submission time (see
Submitting a PyTorch Job below).

It is recommended that your python script accept as an argument the location of
the training data (e.g.: `--training-data /some/path/to/data`), then you can
flexibly mount the drive to some location and provide it to your script.

<!-- prettier-ignore -->
??? warn "Typical AAW Persistent Volume Claims can only connect to one machine at a time"
    The standard storage type (Azure Disk) used for PVCs on AAW are ReadWriteOnce, meaning that they can only be attached to a single machine at once.  This means that every node in your PyTorchJob needs **its own** PVC with data.  Not impossible, but definitely annoying... You can use an Azure File (which supports multiple connections, e.g.: ReadWriteMany), but it has shown notably slower transfer rates.

<!-- prettier-ignore -->
??? warn "Because of the one-machine-at-a-time limit, jobs can have at most 2 GPU nodes (one master, one worker)"
    You can specify a different PVC for the master and worker nodes in a PyTorchJob, but you **cannot specify a unique PVC for each worker**.  This makes scaling past 1 master + 1 worker impossible

### Downloading Training Data Locally before Training

Rather than expecting training data be available right away, you can download
the training during startup. Examples of how you can achieve this include:

- adding a download files section to the shell script that orchestrates the
  training. For example, that script can be something like:

```bash
# Download data
# From Minio
mc cp my-minio-tenant/my/path/to/training/data /somewhere/local

# From your git repository
# git clone my_repo_with_data /somewhere/local

# From internet
# curl ...

# Run script using downloaded data
my_training_script.py --training-data /somewhere/local
```

- adding a function in your python code that downloads data before training,
  like:

```python
def get_training_data():
	# Do something here that gets your data, like download it from MinIO
	pass

def train():
	pass

def __name__ == "__main__":
	get_training_data()
	train()
```

<!-- prettier-ignore -->
??? danger "MinIO credentials are not automatically injected to PyTorchJobs"
    Unlike Jupyter Notebooks/Kubeflow Pipeline jobs, MinIO credentials are not automatically mounted at `/vault/secrets/minio-*`.  These will be added in future, but at present if you want to download from MinIO you must provide MinIO credentials yourself.  The best/safest way is to use a [Kubernetes Secret](https://kubernetes.io/docs/concepts/configuration/secret/), which can be safely mounted into a container as an environment variable and accessed by your script

### Using a Data Loader that Fetches Data from MinIO (or S3 storage)

It is possible to extend PyTorch/fast.ai
[Data Loaders](https://pytorch.org/docs/stable/data.html) to access data
directly from MinIO (or any other remote data). A (very rough) example is
[ImageListS3](https://github.com/ca-scribner/fastai-extensions/blob/master/fastaiextensions/image_list_s3.py),
which is an extension of the fast.ai `ImageList` loader. This implementation is
a naive approach that only allows single-threaded data access due to how `boto3`
works, but it might help as inspiration.

In general, while it might sound like training based on MinIO data rather than
local data would be slower, it doesn't have to be. Data loaders allow for
parallelizing data fetching, so while fetching a single image might have slower
throughput, the loader may have N workers fetching images at the same time that
could achieve the same overall throughput as a local disk. Testing by the user
is needed here as this is very problem specific.

## Submitting a PyTorch Job

Submission of a PyTorch Job is done using a Kubernetes `yaml` file and
`kubectl create`. For example, the following will create a training job with one
master and one worker using the
[MNIST](https://github.com/kubeflow/pytorch-operator/tree/master/examples/mnist)
example described above. This takes advantage of a Docker image that has already
been built and loaded into AAW. Annotations are added below for key points.

```
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "pytorch-dist-mnist-gloo"   # <--- Note 1
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
            - name: pytorch
              image: k8scc01covidacr.azurecr.io/scribner-pytorch_dist_mnist_gloo:latest   # <--- Note 2
              args: ["--backend", "gloo"]   # <--- Note 3
              # Comment out the below resources to use the CPU.
              resources:
                limits:
                  nvidia.com/gpu: 1   # <--- Note 4
          imagePullSecrets:
            - name: image-pull-secret   # <--- Note 5

    Worker:
      replicas: 1  # <--- Note 6
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
            - name: pytorch
              image: k8scc01covidacr.azurecr.io/scribner-pytorch_dist_mnist_gloo:latest   # <--- Note 2
              args: ["--backend", "gloo"]   # <--- Note 3
              # Comment out the below resources to use the CPU.
              resources:
                limits:
                  nvidia.com/gpu: 1   # <--- Note 4
          imagePullSecrets:
            - name: image-pull-secret   # <--- Note 5

```

1.  Specify the name of the job being submitted
2.  Specify the Docker image to be used for master and worker. They can be
    different or the same depending on your use case (in this case they are the
    same)
3.       We can add arguments to the `mnist.py` call.  See `mnist.py` for all arguments and the `Dockerfile` for the calling syntax
4.  If we want to use a GPU during training, we add
    ```
    resources:
    	limits:
            nvidia.com/gpu: 1
    ```
    Otherwise, we omit these lines
5.       As we are pulling an image from the AAW Azure Container Registry, we must provide the private credentials that are included automatically in our namespace (these are provided via a [Kubernetes Secret](https://kubernetes.io/docs/concepts/configuration/secret/))
6.       This sets the number of workers.  The total number of nodes will be `1 master + N workers`.

You can submit this job via:

```
kubectl create myfile.yaml
```

and then monitor using:

- List status of all PyTorchJob submissions: `kubectl get pytorchjob`
- See all associated pods:
  `kubectl get pods -l controller-name=pytorch-operator`
- See logs of a given pod (use podname from previous list):
  `kubectl logs -f podname`
- Describe the status of a given pod (use podname from 2nd bullet):
  `kubectl describe podname`

## Extracting Models/Results

_Under construction_

In general, you must either write out a completed model to your mounted PVC (if
using PVCs) or to MinIO. You cannot access the drive of the master/worker nodes
after they have completed.

# Distributed Training in Older PyTorch/fast.ai Versions

Modern versions of PyTorch have very direct ways of converting `Learner`s to
distributed mode. Older versions (eg: PyTorch v1.4.0/fastai 1.0.58) are less
polished. The below code was used to extend a fastai v1.0.58 Learner to have a
more modern `.to_distributed()` API (there may be easier ways but this worked).

```
# Helpers to overcome differences between fastai v2 and v1 apis
from fastai.basic_data import DataBunch
from fastai.vision import NormType, SplitFuncOrIdxList, nn
from typing import Callable, Optional, Tuple, Union, Any

from fastai.vision.learner import cnn_config, create_body, to_device, apply_init
from fastai import distributed
from fastai.core import ifnone
def unet_learner_distributed(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=None, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, cut:Union[int,Callable]=None, **learn_kwargs:Any)->distributed.Learner:
    """
    Build Unet learner from `data` and `arch`.

    Modified from fastai.vision.learner to use distributed.Learner instead.
    """

    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    model = to_device(models.unet.DynamicUnet(body, n_classes=data.c, img_size=size, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle), data.device)
    learn = distributed.Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))

    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)

    # distributed.Learner has extra .to_distributed() method, but IS NOT distributed yet
    # We can notice this in the worker stdout logs.  if we don't call .to_distributed(),
    # both worker and master will show loss values, etc, per epoch.  If we call this here,
    # the detailed epoch statements are only written on the master logs
    # WARNING: Uses hard coded device (eg: expects config of 1 GPU per node)
    print("Converting Learner to distributed Learner", flush=True)
    learn = learn.to_distributed(0)

    return learn
```

# Launching PyTorch Jobs from a Kubeflow Pipeline

This is planned but not yet available. It is an
[outstanding issue](https://github.com/kubeflow/pytorch-operator/issues/190) for
the operator and Kubeflow Pipelines, with no current development. The effort is
not so great, however (there is prior art from the TFJob/Katib launchers, and
suggestions like [here](https://github.com/kubeflow/pipelines/issues/3445), and
we may develop it ourselves in future. Plans from the main Kubeflow Pipelines
team for this sort of development are
[here](https://github.com/kubeflow/common/blob/master/ROADMAP.md)

# Quirks

Some miscellaneous quirks or unexpected behaviours using DDP.

## Effective Batch Size

When using DDP, each GPU is fed a batch of `batch_size` for each training
iteration. This means that for each training iteration, your
`effective_batch_size` is `batch_size * N_GPU`. For example, if you train using
128 images with a `batch_size`=32, you would see:

- 1GPU (not parallel): total images viewed per step=32, steps per epoch=4,
  images viewed per epoch= 128
- 2GPU: total images viewed per step=64 (32 per GPU), steps per epoch=2, images
  viewed per epoch= 128
- 4GPU: total images viewed per step=128 (32 per GPU), steps per epoch=1, images
  viewed per epoch= 128

This means that as you add GPUs you can likely scale your `learning_rate`
because each step during training is seeing your `effective_batch_size`. How to
set `learning_rate` is beyond the scope here, but a first guess is to scale it
proportionally to `effective_batch_size` (so if you go from 2GPU to 4GPU
(doubling your `effective_batch_size`), double your `learning_rate`).
