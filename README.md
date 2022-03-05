# Experimenting with DNN Pruning

- This is a tutorial/exercise for getting started with the compiler orchestrator project.

As mentioned in the project description:

> Performance optimizers for deep learning models are a major part of these tools,
> but we also include tools for other goals in other domains.

For the purpose of this task, let's look at one such performance optimizer for DNNs.
(Specifically, performance here refers to latency/throughput of inference.)

The goal is for you to get familiar with the practices of _DNN pruning_
and all the concepts that leads to it.

## Environment Setup

We'll use a few python packages in this project that require some setup.
The use of a virtual-env management tool, such as `conda` is highly encouraged.
- You can get familiar with conda
  from [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)
  or [here](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf).

To install the package used for DNN pruning:

1. Create a virtual environment and activate it, if you're using conda.
   - When creating the virtual environment we recommend `python=3.8`.

     If you're using conda, `conda create -n prune python=3.8` will do.

1. From the project directory, run this command:
   ```
   cd yapt/
   pip install -e .
   ```

## Tasks

- There's no need to write down answer to all the questions.
  These are merely a guidance for you to know where to look at.
  The [Deliverable](#deliverable) section shows what to deliver as the result of this exercise.

- There are many questions below to assist your understanding,
  but the actual amount of code needed is small.
  These 2 coding tasks ([**]) below should together take less than 20 lines.

### Understanding DNN Filter Pruning

Filter pruning is a kind of DNN pruning technique applied to Convolution layers
that reduces the size and improves the performance of DNN.

Familiarize yourself with filter pruning by finding out the answer to the following questions:

1. The weight of a convolution layer is a 4-dimensional tensor.
   Which dimensions are the filter dimension / channel dimension? (Why are they called so?)
   - Figure 2 of ["Pruning Filter in Filter", Meng et. al.](https://proceedings.nips.cc/paper/2020/file/ccb1d45fb76f7c5a0bf619f979c6cf36-Paper.pdf) may help.

1. When filters are removed from a layer, in which way is the output of that layer affected?
   Therefore, to make sure the DNN works after pruning,
   what do we also need to do to other layers in the DNN?
   - If you're not sure about this, hold on till you see this in effect in the next section.

### Filter Pruning on ResNet-18

Check out `prune_resnet18.py` at the project root directory.

1. Have a look at the architecture of ResNet-18.
   - Run function `draw_resnet18` in `prune_resnet18.py`. It will generate a `resnet18.png`
     which shows the architecture of the DNN as a control flow graph (**CFG**) where each box is an operation.
   - The image shows the name that PyTorch assigns to each layer (such as "layer1.0.conv1").

1. Prune ResNet-18 with fixed prune ratio.
   - Run function `prune_resnet18_fix_ratio`. This will print out a ResNet-18
     where each Convolution layer has around 20% of its filters removed.
   - Continuing the question in previous section, what else is removed apart from convolution filters?
     Why?
     - You can also print out the shape of weights of each layer to check this.

1. [**] Write your own code to prune ResNet-18 in different ways.
   - In the CFG of ResNet-18, find the convolution layer "layer4.1.conv1".

   - We've just used `FixedPruneRatioPruner` to prune the same percentage of filters from each layer.
     Now imitating the function [`FixedPruneRatioPruner.select_filters`](./yapt/yapt/struct_pruning/pruning.py#L337),
     implement your own `OneLayerPruner.select_filters` to prune only **"layer4.1.conv1"** by 20%.

     A code outline is already given [here](./prune_resnet18.py#L36).

     Feel free to modify `FixedPruneRatioPruner` (or any other code) inplace
     to understand its input/output.

1. [**] Prune groups.
   - Is it possible to only remove filters from the first Conv layer "conv1"?
     (Does your `OneLayerPruner` work for "conv1"?)

     If not, which layers' filters must also be pruned? Why?
     
     Hint: ResNet-18 has so-called skip connections; you can see these in the CFG.

   - Fill in [`FirstGroupPruner.select_filters`](./prune_resnet18.py#L51) to prune all these layers
     by 20%.

## Deliverable

- Send `prune_resnet18.py` with your code filled in if you finish both.

- If you feel stuck or uncertain, you're welcome to also send a document
  with the question you have problems with and your understanding.
