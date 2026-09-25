# PyTorch Full Practical API Reference

**Purpose:** A compact reference for the PyTorch APIs most relevant to deep learning, computer vision, NLP, sequence models, transformers, training, GPU execution, and model deployment.

**Format of every entry:** Syntax → What it does → Parameters → Output.

**Excluded by design:** tutorials, tiny examples, common-mistake sections, and long conceptual explanations.

**Version baseline:** PyTorch 2.14 documentation. API signatures can change between releases; the official documentation is authoritative for release-specific details.

## Contents
- 1. Tensor Creation
- 2. Tensor Shape, Type and Movement
- 3. Shape Operations
- 4. Mathematical and Reduction Operations
- 5. Matrix and Linear Algebra
- 6. torch.nn — Core Modules
- 7. torch.nn — Linear and Convolution
- 8. torch.nn — Activations
- 9. torch.nn — Pooling, Flattening and Normalization
- 10. torch.nn — Dropout and Regularization
- 11. torch.nn — Sequence Models
- 12. torch.nn — Attention and Transformers
- 13. torch.nn — Loss Functions
- 14. torch.nn.functional
- 15. torch.nn.init
- 16. torch.optim
- 17. Optimizer Methods and Schedulers
- 18. torch.utils.data
- 19. torch.autograd
- 20. CUDA, Devices and AMP
- 21. Serialization and Model State
- 22. Compilation and Performance
- 23. Gradient Utilities
- 24. RNN Utilities
- 25. Linear Algebra and FFT
- 26. Reproducibility and Randomness
- 27. torchvision — Transforms and Models
- 28. Module Control and Training

# 1. Tensor Creation
## `torch.tensor`

**Syntax**
```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

**What it does**  
Creates a tensor from tensor-like data.

**Parameters**
- `data` — source data.
- `dtype` — tensor data type.
- `device` — destination device.
- `requires_grad` — autograd tracking.
- `pin_memory` — pinned CPU memory.

**Output**  
`torch.Tensor`.

## `torch.as_tensor`

**Syntax**
```python
torch.as_tensor(data, dtype=None, device=None)
```

**What it does**  
Converts data to a tensor, avoiding a copy when possible.

**Parameters**
- `data` — source data.
- `dtype` — optional dtype.
- `device` — optional device.

**Output**  
`torch.Tensor`.

## `torch.from_numpy`

**Syntax**
```python
torch.from_numpy(ndarray)
```

**What it does**  
Creates a tensor sharing memory with a NumPy array.

**Parameters**
- `ndarray` — supported NumPy array.

**Output**  
`torch.Tensor`.

## `torch.zeros`

**Syntax**
```python
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Creates a tensor filled with zeros.

**Parameters**
- `size` — dimensions.
- `out` — destination tensor.
- `dtype` — data type.
- `layout` — layout.
- `device` — device.
- `requires_grad` — autograd tracking.

**Output**  
Tensor of requested shape filled with zero.

## `torch.ones`

**Syntax**
```python
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Creates a tensor filled with ones.

**Parameters**
- `size` — dimensions.
- `out` — destination tensor.
- `dtype` — data type.
- `layout` — layout.
- `device` — device.
- `requires_grad` — autograd tracking.

**Output**  
Tensor filled with one.

## `torch.empty`

**Syntax**
```python
torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**What it does**  
Allocates a tensor without initializing its values.

**Parameters**
- `size` — dimensions.
- `out` — destination.
- `dtype` — data type.
- `layout` — layout.
- `device` — device.
- `requires_grad` — autograd tracking.
- `pin_memory` — pinned CPU memory.

**Output**  
Uninitialized tensor.

## `torch.full`

**Syntax**
```python
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Creates a tensor filled with a specified value.

**Parameters**
- `size` — dimensions.
- `fill_value` — fill value.
- `out`, `dtype`, `layout`, `device`, `requires_grad` — output settings.

**Output**  
Tensor filled with `fill_value`.

## `torch.arange`

**Syntax**
```python
torch.arange(start=0, end, step=1, *, dtype=None, device=None, requires_grad=False)
```

**What it does**  
Creates evenly spaced values using a step.

**Parameters**
- `start` — starting value.
- `end` — exclusive end.
- `step` — spacing.
- `dtype`, `device`, `requires_grad` — output settings.

**Output**  
1-D tensor.

## `torch.linspace`

**Syntax**
```python
torch.linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False)
```

**What it does**  
Creates evenly spaced values over a closed interval.

**Parameters**
- `start` — first value.
- `end` — last value.
- `steps` — number of values.
- `dtype`, `device`, `requires_grad` — output settings.

**Output**  
1-D tensor.

## `torch.eye`

**Syntax**
```python
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Creates an identity-like matrix.

**Parameters**
- `n` — rows.
- `m` — columns.
- Remaining arguments configure output.

**Output**  
2-D tensor.

## `torch.rand`

**Syntax**
```python
torch.rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Samples from a uniform distribution on `[0,1)`.

**Parameters**
- `size` — dimensions.
- `generator` — RNG.
- `out`, `dtype`, `layout`, `device`, `requires_grad` — output settings.

**Output**  
Random tensor.

## `torch.randn`

**Syntax**
```python
torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Samples from a standard normal distribution.

**Parameters**
- `size` — dimensions.
- `generator` — RNG.
- `out`, `dtype`, `layout`, `device`, `requires_grad` — output settings.

**Output**  
Random tensor.

## `torch.randint`

**Syntax**
```python
torch.randint(low, high, size, *, generator=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**What it does**  
Samples integer values from `[low, high)`.

**Parameters**
- `low` — inclusive lower bound.
- `high` — exclusive upper bound.
- `size` — dimensions.
- Remaining arguments configure output.

**Output**  
Integer tensor.


# 2. Tensor Shape, Type and Movement
## `Tensor.shape`

**Syntax**
```python
tensor.shape
```

**What it does**  
Returns tensor dimensions.

**Parameters**
None.

**Output**  
`torch.Size`.

## `Tensor.ndim`

**Syntax**
```python
tensor.ndim
```

**What it does**  
Returns number of dimensions.

**Parameters**
None.

**Output**  
Integer.

## `Tensor.numel`

**Syntax**
```python
tensor.numel()
```

**What it does**  
Returns total number of elements.

**Parameters**
None.

**Output**  
Integer.

## `Tensor.element_size`

**Syntax**
```python
tensor.element_size()
```

**What it does**  
Returns bytes per element.

**Parameters**
None.

**Output**  
Integer.

## `Tensor.dtype`

**Syntax**
```python
tensor.dtype
```

**What it does**  
Returns tensor data type.

**Parameters**
None.

**Output**  
`torch.dtype`.

## `Tensor.device`

**Syntax**
```python
tensor.device
```

**What it does**  
Returns tensor device.

**Parameters**
None.

**Output**  
`torch.device`.

## `Tensor.to`

**Syntax**
```python
tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

**What it does**  
Moves or casts a tensor.

**Parameters**
- `device` — destination device.
- `dtype` — destination dtype.
- `non_blocking` — asynchronous transfer when supported.
- `copy` — force copy.
- `memory_format` — memory layout.

**Output**  
Converted tensor.

## `Tensor.cpu`

**Syntax**
```python
tensor.cpu()
```

**What it does**  
Returns tensor on CPU.

**Parameters**
None.

**Output**  
CPU tensor.

## `Tensor.cuda`

**Syntax**
```python
tensor.cuda(device=None, non_blocking=False)
```

**What it does**  
Returns tensor on CUDA device.

**Parameters**
- `device` — CUDA device.
- `non_blocking` — asynchronous transfer when supported.

**Output**  
CUDA tensor.

## `Tensor.numpy`

**Syntax**
```python
tensor.numpy(force=False)
```

**What it does**  
Converts a CPU tensor to a NumPy array when supported.

**Parameters**
- `force` — permits a copy/detach path when required.

**Output**  
NumPy array.

## `Tensor.item`

**Syntax**
```python
tensor.item()
```

**What it does**  
Extracts a single-element tensor as a Python scalar.

**Parameters**
Tensor must contain one element.

**Output**  
Python number.

## `Tensor.detach`

**Syntax**
```python
tensor.detach()
```

**What it does**  
Returns a tensor detached from the autograd graph.

**Parameters**
None.

**Output**  
Detached tensor.

## `Tensor.clone`

**Syntax**
```python
tensor.clone(*, memory_format=torch.preserve_format)
```

**What it does**  
Returns a copy of the tensor.

**Parameters**
- `memory_format` — output memory format.

**Output**  
Copied tensor.

## `Tensor.contiguous`

**Syntax**
```python
tensor.contiguous(memory_format=torch.contiguous_format)
```

**What it does**  
Returns a contiguous tensor.

**Parameters**
- `memory_format` — desired layout.

**Output**  
Contiguous tensor.


# 3. Shape Operations
## `torch.reshape`

**Syntax**
```python
torch.reshape(input, shape)
```

**What it does**  
Returns a tensor with the requested shape.

**Parameters**
- `input` — input tensor.
- `shape` — target shape.

**Output**  
Tensor with requested shape.

## `Tensor.view`

**Syntax**
```python
tensor.view(*shape)
```

**What it does**  
Returns a view with a different shape when memory layout permits.

**Parameters**
- `shape` — target shape.

**Output**  
Tensor view.

## `torch.flatten`

**Syntax**
```python
torch.flatten(input, start_dim=0, end_dim=-1)
```

**What it does**  
Flattens a range of dimensions.

**Parameters**
- `input` — tensor.
- `start_dim` — first dimension.
- `end_dim` — last dimension.

**Output**  
Tensor with selected dimensions flattened.

## `torch.squeeze`

**Syntax**
```python
torch.squeeze(input, dim=None)
```

**What it does**  
Removes dimensions of size one.

**Parameters**
- `input` — tensor.
- `dim` — optional dimensions.

**Output**  
Tensor with singleton dimensions removed.

## `torch.unsqueeze`

**Syntax**
```python
torch.unsqueeze(input, dim)
```

**What it does**  
Adds a dimension of size one.

**Parameters**
- `input` — tensor.
- `dim` — insertion dimension.

**Output**  
Tensor with one additional dimension.

## `torch.permute`

**Syntax**
```python
torch.permute(input, dims)
```

**What it does**  
Reorders dimensions.

**Parameters**
- `input` — tensor.
- `dims` — dimension permutation.

**Output**  
View with reordered dimensions.

## `torch.transpose`

**Syntax**
```python
torch.transpose(input, dim0, dim1)
```

**What it does**  
Swaps two dimensions.

**Parameters**
- `input` — tensor.
- `dim0`, `dim1` — dimensions to swap.

**Output**  
Tensor with dimensions exchanged.

## `torch.cat`

**Syntax**
```python
torch.cat(tensors, dim=0, *, out=None)
```

**What it does**  
Concatenates tensors along an existing dimension.

**Parameters**
- `tensors` — sequence of tensors.
- `dim` — concatenation dimension.
- `out` — destination.

**Output**  
Concatenated tensor.

## `torch.stack`

**Syntax**
```python
torch.stack(tensors, dim=0, *, out=None)
```

**What it does**  
Combines tensors along a new dimension.

**Parameters**
- `tensors` — equal-shaped tensors.
- `dim` — new dimension.
- `out` — destination.

**Output**  
Tensor with one extra dimension.

## `torch.chunk`

**Syntax**
```python
torch.chunk(input, chunks, dim=0)
```

**What it does**  
Splits a tensor into chunks.

**Parameters**
- `input` — tensor.
- `chunks` — requested number.
- `dim` — split dimension.

**Output**  
Tuple of tensors.

## `torch.split`

**Syntax**
```python
torch.split(tensor, split_size_or_sections, dim=0)
```

**What it does**  
Splits by fixed size or explicit sections.

**Parameters**
- `tensor` — input.
- `split_size_or_sections` — chunk specification.
- `dim` — split dimension.

**Output**  
Tuple of tensors.


# 4. Mathematical and Reduction Operations
## `torch.sum`

**Syntax**
```python
torch.sum(input, dim=None, keepdim=False, *, dtype=None)
```

**What it does**  
Sums tensor elements.

**Parameters**
- `input` — tensor.
- `dim` — reduction dimensions.
- `keepdim` — retain reduced dimensions.
- `dtype` — accumulation/output dtype.

**Output**  
Tensor of sums.

## `torch.mean`

**Syntax**
```python
torch.mean(input, dim=None, keepdim=False, *, dtype=None)
```

**What it does**  
Computes arithmetic mean.

**Parameters**
- `input` — tensor.
- `dim` — reduction dimensions.
- `keepdim` — retain dimensions.
- `dtype` — output dtype.

**Output**  
Tensor of means.

## `torch.max`

**Syntax**
```python
torch.max(input, dim=None, keepdim=False, *, out=None)
```

**What it does**  
Computes maximum values, optionally along a dimension.

**Parameters**
- `input` — tensor.
- `dim` — reduction dimension.
- `keepdim` — retain dimension.
- `out` — destination.

**Output**  
Maximum tensor; with dimension reduction, values and indices.

## `torch.min`

**Syntax**
```python
torch.min(input, dim=None, keepdim=False, *, out=None)
```

**What it does**  
Computes minimum values.

**Parameters**
- `input` — tensor.
- `dim` — reduction dimension.
- `keepdim` — retain dimension.
- `out` — destination.

**Output**  
Minimum tensor; with dimension reduction, values and indices.

## `torch.argmax`

**Syntax**
```python
torch.argmax(input, dim=None, keepdim=False)
```

**What it does**  
Returns indices of maximum values.

**Parameters**
- `input` — tensor.
- `dim` — optional dimension.
- `keepdim` — retain dimension.

**Output**  
Long tensor of indices.

## `torch.argmin`

**Syntax**
```python
torch.argmin(input, dim=None, keepdim=False)
```

**What it does**  
Returns indices of minimum values.

**Parameters**
- `input` — tensor.
- `dim` — optional dimension.
- `keepdim` — retain dimension.

**Output**  
Long tensor of indices.

## `torch.abs`

**Syntax**
```python
torch.abs(input, *, out=None)
```

**What it does**  
Computes absolute values.

**Parameters**
- `input` — tensor.
- `out` — destination.

**Output**  
Tensor.

## `torch.sqrt`

**Syntax**
```python
torch.sqrt(input, *, out=None)
```

**What it does**  
Computes square roots.

**Parameters**
- `input` — tensor.
- `out` — destination.

**Output**  
Tensor.

## `torch.exp`

**Syntax**
```python
torch.exp(input, *, out=None)
```

**What it does**  
Computes exponentials.

**Parameters**
- `input` — tensor.
- `out` — destination.

**Output**  
Tensor.

## `torch.log`

**Syntax**
```python
torch.log(input, *, out=None)
```

**What it does**  
Computes natural logarithms.

**Parameters**
- `input` — tensor.
- `out` — destination.

**Output**  
Tensor.

## `torch.pow`

**Syntax**
```python
torch.pow(input, exponent, *, out=None)
```

**What it does**  
Raises values to a power.

**Parameters**
- `input` — tensor.
- `exponent` — exponent.
- `out` — destination.

**Output**  
Tensor.

## `torch.clamp`

**Syntax**
```python
torch.clamp(input, min=None, max=None, *, out=None)
```

**What it does**  
Restricts values to a range.

**Parameters**
- `input` — tensor.
- `min` — lower bound.
- `max` — upper bound.
- `out` — destination.

**Output**  
Tensor with bounded values.

## `torch.where`

**Syntax**
```python
torch.where(condition, input, other, *, out=None)
```

**What it does**  
Selects values element-wise according to a boolean condition.

**Parameters**
- `condition` — boolean tensor.
- `input` — values where condition is true.
- `other` — values where false.
- `out` — destination.

**Output**  
Selected tensor.

## `torch.logical_and`

**Syntax**
```python
torch.logical_and(input, other, *, out=None)
```

**What it does**  
Computes element-wise logical AND.

**Parameters**
- `input`, `other` — boolean-compatible tensors.
- `out` — destination.

**Output**  
Boolean tensor.

## `torch.logical_or`

**Syntax**
```python
torch.logical_or(input, other, *, out=None)
```

**What it does**  
Computes element-wise logical OR.

**Parameters**
- `input`, `other` — boolean-compatible tensors.
- `out` — destination.

**Output**  
Boolean tensor.


# 5. Matrix and Linear Algebra
## `torch.matmul`

**Syntax**
```python
torch.matmul(input, other, *, out=None)
```

**What it does**  
Performs matrix multiplication with broadcasting for higher dimensions.

**Parameters**
- `input` — first operand.
- `other` — second operand.
- `out` — destination.

**Output**  
Matrix product tensor.

## `torch.mm`

**Syntax**
```python
torch.mm(input, mat2, *, out=None)
```

**What it does**  
Matrix multiplication for two 2-D tensors.

**Parameters**
- `input`, `mat2` — matrices.
- `out` — destination.

**Output**  
2-D matrix.

## `torch.bmm`

**Syntax**
```python
torch.bmm(input, mat2, *, out=None)
```

**What it does**  
Batch matrix multiplication for 3-D tensors.

**Parameters**
- `input`, `mat2` — batches of matrices.
- `out` — destination.

**Output**  
3-D batch of products.

## `torch.einsum`

**Syntax**
```python
torch.einsum(equation, *operands)
```

**What it does**  
Performs generalized Einstein summation.

**Parameters**
- `equation` — summation specification.
- `operands` — input tensors.

**Output**  
Tensor defined by the equation.


# 6. torch.nn — Core Modules
## `nn.Module`

**Syntax**
```python
class Module
```

**What it does**  
Base class for neural-network modules and parameter/buffer registration.

**Parameters**
- Subclass-specific constructor parameters.
- Core methods include `forward`, `parameters`, `state_dict`, `load_state_dict`, `train`, `eval`, `to`.

**Output**  
Module object.

## `nn.Parameter`

**Syntax**
```python
nn.Parameter(data=None, requires_grad=True)
```

**What it does**  
Tensor subclass automatically registered as a module parameter.

**Parameters**
- `data` — underlying tensor.
- `requires_grad` — gradient tracking.

**Output**  
`nn.Parameter`.

## `nn.Sequential`

**Syntax**
```python
nn.Sequential(*args)
```

**What it does**  
Applies child modules sequentially.

**Parameters**
- `*args` — modules or ordered mapping.

**Output**  
Module container.

## `nn.ModuleList`

**Syntax**
```python
nn.ModuleList(modules=None)
```

**What it does**  
Registers submodules in a list.

**Parameters**
- `modules` — iterable of modules.

**Output**  
Module container.

## `nn.ModuleDict`

**Syntax**
```python
nn.ModuleDict(modules=None)
```

**What it does**  
Registers submodules in a dictionary-like container.

**Parameters**
- `modules` — mapping or name/module pairs.

**Output**  
Module container.

## `nn.Identity`

**Syntax**
```python
nn.Identity(*args, **kwargs)
```

**What it does**  
Passes input through unchanged.

**Parameters**
- Constructor arguments are accepted for compatibility and ignored.

**Output**  
Input tensor unchanged.


# 7. torch.nn — Linear and Convolution
## `nn.Linear`

**Syntax**
```python
nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

**What it does**  
Applies an affine transformation.

**Parameters**
- `in_features` — input features.
- `out_features` — output features.
- `bias` — learnable bias.
- `device`, `dtype` — parameter configuration.

**Output**  
Tensor with final dimension `out_features`.

## `nn.Bilinear`

**Syntax**
```python
nn.Bilinear(in1_features, in2_features, out_features, bias=True, device=None, dtype=None)
```

**What it does**  
Applies a bilinear transformation to two inputs.

**Parameters**
- `in1_features` — first feature size.
- `in2_features` — second feature size.
- `out_features` — output size.
- `bias`, `device`, `dtype` — configuration.

**Output**  
Bilinear output tensor.

## `nn.Conv1d`

**Syntax**
```python
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

**What it does**  
Applies 1-D convolution.

**Parameters**
- `in_channels`, `out_channels` — channels.
- `kernel_size` — kernel size.
- `stride` — stride.
- `padding` — padding.
- `dilation` — dilation.
- `groups` — channel groups.
- `bias` — bias.
- `padding_mode` — padding method.
- `device`, `dtype` — configuration.

**Output**  
3-D tensor `[N,C_out,L_out]`.

## `nn.Conv2d`

**Syntax**
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

**What it does**  
Applies 2-D convolution.

**Parameters**
- `in_channels`, `out_channels` — channels/filters.
- `kernel_size` — kernel size.
- `stride` — stride.
- `padding` — padding.
- `dilation` — dilation.
- `groups` — grouped convolution.
- `bias` — bias.
- `padding_mode` — padding method.
- `device`, `dtype` — configuration.

**Output**  
4-D tensor `[N,C_out,H_out,W_out]`.

## `nn.Conv3d`

**Syntax**
```python
nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

**What it does**  
Applies 3-D convolution.

**Parameters**
- `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias`, `padding_mode`, `device`, `dtype`.

**Output**  
5-D tensor `[N,C_out,D_out,H_out,W_out]`.

## `nn.ConvTranspose2d`

**Syntax**
```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
```

**What it does**  
Applies a 2-D transposed convolution.

**Parameters**
- `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `output_padding`, `groups`, `bias`, `dilation`, `padding_mode`, `device`, `dtype`.

**Output**  
4-D output tensor.


# 8. torch.nn — Activations
## `nn.ReLU`

**Syntax**
```python
nn.ReLU(inplace=False)
```

**What it does**  
Applies `max(0,x)` element-wise.

**Parameters**
- `inplace` — modify input in-place.

**Output**  
Same shape as input.

## `nn.LeakyReLU`

**Syntax**
```python
nn.LeakyReLU(negative_slope=0.01, inplace=False)
```

**What it does**  
Applies ReLU with a nonzero negative slope.

**Parameters**
- `negative_slope` — negative-region slope.
- `inplace` — in-place operation.

**Output**  
Same shape.

## `nn.PReLU`

**Syntax**
```python
nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)
```

**What it does**  
Applies PReLU with learnable negative slope(s).

**Parameters**
- `num_parameters` — number of slopes.
- `init` — initial slope.
- `device`, `dtype` — configuration.

**Output**  
Same shape.

## `nn.ELU`

**Syntax**
```python
nn.ELU(alpha=1.0, inplace=False)
```

**What it does**  
Applies Exponential Linear Unit.

**Parameters**
- `alpha` — negative-region scale.
- `inplace` — in-place operation.

**Output**  
Same shape.

## `nn.GELU`

**Syntax**
```python
nn.GELU(approximate='none')
```

**What it does**  
Applies Gaussian Error Linear Unit.

**Parameters**
- `approximate` — approximation method.

**Output**  
Same shape.

## `nn.Sigmoid`

**Syntax**
```python
nn.Sigmoid()
```

**What it does**  
Maps values to `(0,1)`.

**Parameters**
None.

**Output**  
Same shape.

## `nn.Tanh`

**Syntax**
```python
nn.Tanh()
```

**What it does**  
Applies hyperbolic tangent.

**Parameters**
None.

**Output**  
Same shape.

## `nn.Softmax`

**Syntax**
```python
nn.Softmax(dim=None)
```

**What it does**  
Normalizes values with softmax along a dimension.

**Parameters**
- `dim` — normalization dimension.

**Output**  
Same shape.

## `nn.LogSoftmax`

**Syntax**
```python
nn.LogSoftmax(dim=None)
```

**What it does**  
Computes log-softmax along a dimension.

**Parameters**
- `dim` — normalization dimension.

**Output**  
Same shape.


# 9. torch.nn — Pooling, Flattening and Normalization
## `nn.MaxPool2d`

**Syntax**
```python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

**What it does**  
Applies 2-D max pooling.

**Parameters**
- `kernel_size` — window.
- `stride` — stride.
- `padding` — padding.
- `dilation` — window dilation.
- `return_indices` — return indices.
- `ceil_mode` — output rounding.

**Output**  
Pooled tensor; optionally indices.

## `nn.AvgPool2d`

**Syntax**
```python
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

**What it does**  
Applies 2-D average pooling.

**Parameters**
- `kernel_size`, `stride`, `padding` — geometry.
- `ceil_mode` — rounding.
- `count_include_pad` — include padding.
- `divisor_override` — averaging divisor.

**Output**  
Pooled tensor.

## `nn.AdaptiveAvgPool2d`

**Syntax**
```python
nn.AdaptiveAvgPool2d(output_size)
```

**What it does**  
Produces a specified spatial output size by adaptive average pooling.

**Parameters**
- `output_size` — target height and width.

**Output**  
Tensor with requested spatial size.

## `nn.AdaptiveMaxPool2d`

**Syntax**
```python
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
```

**What it does**  
Produces a specified spatial output size by adaptive max pooling.

**Parameters**
- `output_size` — target size.
- `return_indices` — return indices.

**Output**  
Pooled tensor; optionally indices.

## `nn.Flatten`

**Syntax**
```python
nn.Flatten(start_dim=1, end_dim=-1)
```

**What it does**  
Flattens selected dimensions.

**Parameters**
- `start_dim` — first dimension.
- `end_dim` — last dimension.

**Output**  
Tensor with selected dimensions flattened.

## `nn.Unflatten`

**Syntax**
```python
nn.Unflatten(dim, unflattened_size)
```

**What it does**  
Expands one dimension into multiple dimensions.

**Parameters**
- `dim` — dimension to expand.
- `unflattened_size` — new dimensions.

**Output**  
Reshaped tensor.

## `nn.BatchNorm1d`

**Syntax**
```python
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

**What it does**  
Batch-normalizes 1-D features.

**Parameters**
- `num_features` — feature/channel count.
- `eps` — numerical stability.
- `momentum` — running-stat update.
- `affine` — learn scale/bias.
- `track_running_stats` — keep running statistics.
- `device`, `dtype`.

**Output**  
Same shape as input.

## `nn.BatchNorm2d`

**Syntax**
```python
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

**What it does**  
Batch-normalizes 2-D feature maps.

**Parameters**
- Same parameters as `BatchNorm1d`; `num_features` is channels.

**Output**  
Same shape as input.

## `nn.LayerNorm`

**Syntax**
```python
nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None)
```

**What it does**  
Normalizes over the final specified dimensions.

**Parameters**
- `normalized_shape` — normalized dimensions.
- `eps` — numerical stability.
- `elementwise_affine` — learn scale/bias.
- `bias` — learn bias.
- `device`, `dtype`.

**Output**  
Same shape as input.

## `nn.GroupNorm`

**Syntax**
```python
nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None)
```

**What it does**  
Normalizes channels in groups.

**Parameters**
- `num_groups` — groups.
- `num_channels` — channels.
- `eps` — numerical stability.
- `affine` — learn scale/bias.
- `device`, `dtype`.

**Output**  
Same shape as input.


# 10. torch.nn — Dropout and Regularization
## `nn.Dropout`

**Syntax**
```python
nn.Dropout(p=0.5, inplace=False)
```

**What it does**  
Randomly zeros individual elements during training.

**Parameters**
- `p` — drop probability.
- `inplace` — in-place operation.

**Output**  
Same shape.

## `nn.Dropout1d`

**Syntax**
```python
nn.Dropout1d(p=0.5, inplace=False)
```

**What it does**  
Randomly zeros entire channels for 1-D feature maps.

**Parameters**
- `p` — probability.
- `inplace` — in-place operation.

**Output**  
Same shape.

## `nn.Dropout2d`

**Syntax**
```python
nn.Dropout2d(p=0.5, inplace=False)
```

**What it does**  
Randomly zeros entire channels for 2-D feature maps.

**Parameters**
- `p` — probability.
- `inplace` — in-place operation.

**Output**  
Same shape.

## `nn.Dropout3d`

**Syntax**
```python
nn.Dropout3d(p=0.5, inplace=False)
```

**What it does**  
Randomly zeros entire channels for 3-D feature maps.

**Parameters**
- `p` — probability.
- `inplace` — in-place operation.

**Output**  
Same shape.


# 11. torch.nn — Sequence Models
## `nn.RNN`

**Syntax**
```python
nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
```

**What it does**  
Vanilla recurrent neural network.

**Parameters**
- `input_size` — input features.
- `hidden_size` — hidden size.
- `num_layers` — recurrent layers.
- `nonlinearity` — `tanh` or `relu`.
- `bias` — bias parameters.
- `batch_first` — batch dimension first.
- `dropout` — inter-layer dropout.
- `bidirectional` — two directions.
- `device`, `dtype`.

**Output**  
Tuple `(output, h_n)`.

## `nn.LSTM`

**Syntax**
```python
nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
```

**What it does**  
Long Short-Term Memory recurrent network.

**Parameters**
- `input_size`, `hidden_size`, `num_layers`, `bias`, `batch_first`, `dropout`, `bidirectional`.
- `proj_size` — projection size.
- `device`, `dtype`.

**Output**  
Tuple `(output, (h_n, c_n))`.

## `nn.GRU`

**Syntax**
```python
nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
```

**What it does**  
Gated Recurrent Unit network.

**Parameters**
- `input_size`, `hidden_size`, `num_layers`, `bias`, `batch_first`, `dropout`, `bidirectional`, `device`, `dtype`.

**Output**  
Tuple `(output, h_n)`.

## `nn.Embedding`

**Syntax**
```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
```

**What it does**  
Maps integer indices to learnable dense vectors.

**Parameters**
- `num_embeddings` — table size.
- `embedding_dim` — vector size.
- `padding_idx` — padding index.
- `max_norm` — norm cap.
- `norm_type` — norm order.
- `scale_grad_by_freq` — frequency scaling.
- `sparse` — sparse gradients.
- `_weight` — supplied weights.
- `_freeze` — freeze supplied weights.
- `device`, `dtype`.

**Output**  
Embedding tensor.


# 12. torch.nn — Attention and Transformers
## `nn.MultiheadAttention`

**Syntax**
```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
```

**What it does**  
Computes multi-head attention over query, key and value.

**Parameters**
- `embed_dim` — embedding size.
- `num_heads` — attention heads.
- `dropout` — attention dropout.
- `bias` — projection bias.
- `add_bias_kv` — bias vectors.
- `add_zero_attn` — add zero attention.
- `kdim`, `vdim` — key/value dimensions.
- `batch_first` — layout.
- `device`, `dtype`.

**Output**  
Attention output and optionally attention weights.

## `nn.TransformerEncoderLayer`

**Syntax**
```python
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
```

**What it does**  
One Transformer encoder layer.

**Parameters**
- `d_model` — model dimension.
- `nhead` — heads.
- `dim_feedforward` — FFN dimension.
- `dropout` — dropout.
- `activation` — activation.
- `layer_norm_eps` — epsilon.
- `batch_first` — layout.
- `norm_first` — pre-normalization.
- `bias` — bias.
- `device`, `dtype`.

**Output**  
Encoded sequence tensor.

## `nn.TransformerEncoder`

**Syntax**
```python
nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
```

**What it does**  
Stacks Transformer encoder layers.

**Parameters**
- `encoder_layer` — layer template.
- `num_layers` — depth.
- `norm` — optional final norm.
- `enable_nested_tensor` — nested tensor support.
- `mask_check` — mask validation.

**Output**  
Encoded sequence tensor.

## `nn.TransformerDecoderLayer`

**Syntax**
```python
nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
```

**What it does**  
One Transformer decoder layer with self- and cross-attention.

**Parameters**
- Same core configuration as encoder layer.

**Output**  
Decoded sequence tensor.

## `nn.TransformerDecoder`

**Syntax**
```python
nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
```

**What it does**  
Stacks Transformer decoder layers.

**Parameters**
- `decoder_layer` — layer template.
- `num_layers` — depth.
- `norm` — optional final norm.

**Output**  
Decoded sequence tensor.

## `nn.Transformer`

**Syntax**
```python
nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
```

**What it does**  
Full encoder-decoder Transformer.

**Parameters**
- `d_model` — model dimension.
- `nhead` — heads.
- `num_encoder_layers`, `num_decoder_layers` — depth.
- `dim_feedforward` — FFN size.
- `dropout` — dropout.
- `activation` — activation.
- `custom_encoder`, `custom_decoder` — replacements.
- `layer_norm_eps`, `batch_first`, `norm_first`, `bias`, `device`, `dtype`.

**Output**  
Transformer decoder output.


# 13. torch.nn — Loss Functions
## `nn.MSELoss`

**Syntax**
```python
nn.MSELoss(size_average=None, reduce=None, reduction='mean')
```

**What it does**  
Computes mean squared error.

**Parameters**
- `reduction` — `none`, `mean`, or `sum`.
- `size_average`, `reduce` — legacy arguments.

**Output**  
Loss tensor.

## `nn.L1Loss`

**Syntax**
```python
nn.L1Loss(size_average=None, reduce=None, reduction='mean')
```

**What it does**  
Computes mean absolute error.

**Parameters**
- `reduction` — reduction mode.
- `size_average`, `reduce` — legacy arguments.

**Output**  
Loss tensor.

## `nn.SmoothL1Loss`

**Syntax**
```python
nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
```

**What it does**  
Computes smooth L1 loss.

**Parameters**
- `reduction` — reduction mode.
- `beta` — transition point.
- `size_average`, `reduce` — legacy.

**Output**  
Loss tensor.

## `nn.CrossEntropyLoss`

**Syntax**
```python
nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
```

**What it does**  
Computes cross-entropy loss for classification.

**Parameters**
- `weight` — class weights.
- `ignore_index` — ignored target.
- `reduction` — reduction.
- `label_smoothing` — smoothing.
- `size_average`, `reduce` — legacy.

**Output**  
Classification loss.

## `nn.BCELoss`

**Syntax**
```python
nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

**What it does**  
Computes binary cross entropy from probabilities.

**Parameters**
- `weight` — element weights.
- `reduction` — reduction.
- `size_average`, `reduce` — legacy.

**Output**  
Binary cross-entropy loss.

## `nn.BCEWithLogitsLoss`

**Syntax**
```python
nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

**What it does**  
Computes numerically stable BCE directly from logits.

**Parameters**
- `weight` — element weights.
- `reduction` — reduction.
- `pos_weight` — positive-class weighting.
- `size_average`, `reduce` — legacy.

**Output**  
Binary classification loss.

## `nn.NLLLoss`

**Syntax**
```python
nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

**What it does**  
Computes negative log likelihood loss.

**Parameters**
- `weight` — class weights.
- `ignore_index` — ignored target.
- `reduction` — reduction.
- legacy `size_average`, `reduce`.

**Output**  
Loss tensor.

## `nn.KLDivLoss`

**Syntax**
```python
nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
```

**What it does**  
Computes KL divergence.

**Parameters**
- `reduction` — reduction.
- `log_target` — whether target is log-space.
- legacy `size_average`, `reduce`.

**Output**  
Loss tensor.

## `nn.CosineEmbeddingLoss`

**Syntax**
```python
nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

**What it does**  
Measures cosine-similarity relationships between paired inputs.

**Parameters**
- `margin` — negative-pair margin.
- `reduction` — reduction.
- legacy `size_average`, `reduce`.

**Output**  
Loss tensor.


# 14. torch.nn.functional
## `F.linear`

**Syntax**
```python
F.linear(input, weight, bias=None)
```

**What it does**  
Applies an affine transformation.

**Parameters**
- `input` — input.
- `weight` — matrix.
- `bias` — optional bias.

**Output**  
Tensor.

## `F.relu`

**Syntax**
```python
F.relu(input, inplace=False)
```

**What it does**  
Applies ReLU.

**Parameters**
- `input` — tensor.
- `inplace` — in-place operation.

**Output**  
Tensor.

## `F.gelu`

**Syntax**
```python
F.gelu(input, approximate='none')
```

**What it does**  
Applies GELU.

**Parameters**
- `input` — tensor.
- `approximate` — approximation.

**Output**  
Tensor.

## `F.softmax`

**Syntax**
```python
F.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

**What it does**  
Applies softmax.

**Parameters**
- `input` — tensor.
- `dim` — normalization dimension.
- `dtype` — optional computation dtype.

**Output**  
Tensor.

## `F.log_softmax`

**Syntax**
```python
F.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
```

**What it does**  
Applies log-softmax.

**Parameters**
- `input` — tensor.
- `dim` — normalization dimension.
- `dtype` — optional computation dtype.

**Output**  
Tensor.

## `F.dropout`

**Syntax**
```python
F.dropout(input, p=0.5, training=True, inplace=False)
```

**What it does**  
Applies dropout according to training mode.

**Parameters**
- `input` — tensor.
- `p` — drop probability.
- `training` — enable dropout.
- `inplace` — in-place operation.

**Output**  
Tensor.

## `F.conv2d`

**Syntax**
```python
F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

**What it does**  
Applies 2-D convolution.

**Parameters**
- `input`, `weight` — tensors.
- `bias` — optional bias.
- `stride`, `padding`, `dilation`, `groups` — convolution configuration.

**Output**  
4-D convolution output.

## `F.max_pool2d`

**Syntax**
```python
F.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
```

**What it does**  
Applies 2-D max pooling.

**Parameters**
- `input` — tensor.
- `kernel_size`, `stride`, `padding`, `dilation` — pooling geometry.
- `ceil_mode` — rounding.
- `return_indices` — indices.

**Output**  
Pooled tensor; optionally indices.

## `F.avg_pool2d`

**Syntax**
```python
F.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

**What it does**  
Applies 2-D average pooling.

**Parameters**
- `input`, `kernel_size`, `stride`, `padding`, `ceil_mode`, `count_include_pad`, `divisor_override`.

**Output**  
Pooled tensor.

## `F.cross_entropy`

**Syntax**
```python
F.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
```

**What it does**  
Computes cross entropy.

**Parameters**
- `input` — logits.
- `target` — targets.
- `weight` — class weights.
- `ignore_index` — ignored target.
- `reduction` — reduction.
- `label_smoothing` — smoothing.
- legacy `size_average`, `reduce`.

**Output**  
Loss tensor.

## `F.mse_loss`

**Syntax**
```python
F.mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None)
```

**What it does**  
Computes mean squared error.

**Parameters**
- `input`, `target` — tensors.
- `reduction` — reduction.
- `weight` — optional weighting.
- legacy `size_average`, `reduce`.

**Output**  
Loss tensor.

## `F.binary_cross_entropy_with_logits`

**Syntax**
```python
F.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

**What it does**  
Computes stable BCE from logits.

**Parameters**
- `input`, `target` — tensors.
- `weight` — element weights.
- `reduction` — reduction.
- `pos_weight` — positive weighting.
- legacy `size_average`, `reduce`.

**Output**  
Loss tensor.

## `F.normalize`

**Syntax**
```python
F.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
```

**What it does**  
Normalizes vectors along a dimension.

**Parameters**
- `input` — tensor.
- `p` — norm order.
- `dim` — normalization dimension.
- `eps` — numerical stability.
- `out` — destination.

**Output**  
Normalized tensor.


# 15. torch.nn.init
## `nn.init.calculate_gain`

**Syntax**
```python
nn.init.calculate_gain(nonlinearity, param=None)
```

**What it does**  
Returns recommended initialization gain.

**Parameters**
- `nonlinearity` — activation name.
- `param` — optional activation parameter.

**Output**  
Float gain.

## `nn.init.uniform_`

**Syntax**
```python
nn.init.uniform_(tensor, a=0.0, b=1.0, generator=None)
```

**What it does**  
Initializes from a uniform distribution.

**Parameters**
- `tensor` — target.
- `a`, `b` — bounds.
- `generator` — RNG.

**Output**  
Initialized tensor.

## `nn.init.normal_`

**Syntax**
```python
nn.init.normal_(tensor, mean=0.0, std=1.0, generator=None)
```

**What it does**  
Initializes from a normal distribution.

**Parameters**
- `tensor` — target.
- `mean` — mean.
- `std` — standard deviation.
- `generator` — RNG.

**Output**  
Initialized tensor.

## `nn.init.constant_`

**Syntax**
```python
nn.init.constant_(tensor, val)
```

**What it does**  
Fills tensor with a constant.

**Parameters**
- `tensor` — target.
- `val` — constant.

**Output**  
Initialized tensor.

## `nn.init.zeros_`

**Syntax**
```python
nn.init.zeros_(tensor)
```

**What it does**  
Fills tensor with zeros.

**Parameters**
- `tensor` — target.

**Output**  
Initialized tensor.

## `nn.init.ones_`

**Syntax**
```python
nn.init.ones_(tensor)
```

**What it does**  
Fills tensor with ones.

**Parameters**
- `tensor` — target.

**Output**  
Initialized tensor.

## `nn.init.xavier_uniform_`

**Syntax**
```python
nn.init.xavier_uniform_(tensor, gain=1.0, generator=None)
```

**What it does**  
Xavier/Glorot uniform initialization.

**Parameters**
- `tensor` — target.
- `gain` — scaling.
- `generator` — RNG.

**Output**  
Initialized tensor.

## `nn.init.xavier_normal_`

**Syntax**
```python
nn.init.xavier_normal_(tensor, gain=1.0, generator=None)
```

**What it does**  
Xavier/Glorot normal initialization.

**Parameters**
- `tensor` — target.
- `gain` — scaling.
- `generator` — RNG.

**Output**  
Initialized tensor.

## `nn.init.kaiming_uniform_`

**Syntax**
```python
nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
```

**What it does**  
Kaiming/He uniform initialization.

**Parameters**
- `tensor` — target.
- `a` — negative slope.
- `mode` — fan-in/fan-out.
- `nonlinearity` — activation.
- `generator` — RNG.

**Output**  
Initialized tensor.

## `nn.init.kaiming_normal_`

**Syntax**
```python
nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
```

**What it does**  
Kaiming/He normal initialization.

**Parameters**
- `tensor` — target.
- `a`, `mode`, `nonlinearity`, `generator`.

**Output**  
Initialized tensor.


# 16. torch.optim
## `optim.SGD`

**Syntax**
```python
optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)
```

**What it does**  
Stochastic gradient descent with optional momentum.

**Parameters**
- `params` — parameters.
- `lr` — learning rate.
- `momentum` — momentum.
- `dampening` — dampening.
- `weight_decay` — weight decay.
- `nesterov` — Nesterov momentum.
- `maximize` — maximize objective.
- `foreach`, `differentiable`, `fused` — implementation controls.

**Output**  
SGD optimizer.

## `optim.Adam`

**Syntax**
```python
optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False)
```

**What it does**  
Adaptive first/second-moment optimizer.

**Parameters**
- `params`, `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`.
- `foreach`, `maximize`, `capturable`, `differentiable`, `fused`, `decoupled_weight_decay`.

**Output**  
Adam optimizer.

## `optim.AdamW`

**Syntax**
```python
optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=True)
```

**What it does**  
Adam with decoupled weight decay.

**Parameters**
- `params`, `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`.
- `maximize`, `foreach`, `capturable`, `differentiable`, `fused`.

**Output**  
AdamW optimizer.

## `optim.RMSprop`

**Syntax**
```python
optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, *, foreach=None, maximize=False, differentiable=False, capturable=False, fused=False)
```

**What it does**  
Adaptive optimizer based on running squared gradients.

**Parameters**
- `params`, `lr`, `alpha`, `eps`, `weight_decay`, `momentum`, `centered`.
- Implementation controls.

**Output**  
RMSprop optimizer.

## `optim.Adagrad`

**Syntax**
```python
optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach=None, *, maximize=False, differentiable=False, fused=False)
```

**What it does**  
Adaptive optimizer using accumulated squared gradients.

**Parameters**
- `params`, `lr`, `lr_decay`, `weight_decay`, `initial_accumulator_value`, `eps`.
- Implementation controls.

**Output**  
Adagrad optimizer.

## `optim.Adadelta`

**Syntax**
```python
optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, *, foreach=None, maximize=False, differentiable=False, capturable=False)
```

**What it does**  
Adaptive optimizer using exponentially decaying gradient statistics.

**Parameters**
- `params`, `lr`, `rho`, `eps`, `weight_decay`.
- Implementation controls.

**Output**  
Adadelta optimizer.

## `optim.NAdam`

**Syntax**
```python
optim.NAdam(params, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=0.004, *, decoupled_weight_decay=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
```

**What it does**  
Adam-style optimizer with Nesterov momentum.

**Parameters**
- `params`, `lr`, `betas`, `eps`, `weight_decay`, `momentum_decay`.
- Remaining options control decay, execution and differentiation.

**Output**  
NAdam optimizer.

## `optim.RAdam`

**Syntax**
```python
optim.RAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, *, decoupled_weight_decay=False, foreach=None, maximize=False, capturable=False, differentiable=False)
```

**What it does**  
Rectified Adam optimizer.

**Parameters**
- `params`, `lr`, `betas`, `eps`, `weight_decay`.
- Remaining options control decay, execution and differentiation.

**Output**  
RAdam optimizer.


# 17. Optimizer Methods and Schedulers
## `Optimizer.zero_grad`

**Syntax**
```python
optimizer.zero_grad(set_to_none=True)
```

**What it does**  
Clears accumulated gradients.

**Parameters**
- `set_to_none` — use `None` instead of zero tensors.

**Output**  
`None`.

## `Optimizer.step`

**Syntax**
```python
optimizer.step(closure=None)
```

**What it does**  
Updates parameters using optimizer state and gradients.

**Parameters**
- `closure` — optional reevaluation callable.

**Output**  
Usually `None`; optimizer-dependent otherwise.

## `Optimizer.state_dict`

**Syntax**
```python
optimizer.state_dict()
```

**What it does**  
Returns optimizer state and parameter groups.

**Parameters**
None.

**Output**  
Dictionary.

## `Optimizer.load_state_dict`

**Syntax**
```python
optimizer.load_state_dict(state_dict)
```

**What it does**  
Restores optimizer state.

**Parameters**
- `state_dict` — saved optimizer state.

**Output**  
`None`.

## `StepLR`

**Syntax**
```python
StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

**What it does**  
Decays learning rate by `gamma` every `step_size` scheduler steps.

**Parameters**
- `optimizer`, `step_size`, `gamma`, `last_epoch`.

**Output**  
Scheduler.

## `MultiStepLR`

**Syntax**
```python
MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

**What it does**  
Decays learning rate at specified milestones.

**Parameters**
- `optimizer`, `milestones`, `gamma`, `last_epoch`.

**Output**  
Scheduler.

## `ExponentialLR`

**Syntax**
```python
ExponentialLR(optimizer, gamma, last_epoch=-1)
```

**What it does**  
Exponentially decays learning rate.

**Parameters**
- `optimizer`, `gamma`, `last_epoch`.

**Output**  
Scheduler.

## `CosineAnnealingLR`

**Syntax**
```python
CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

**What it does**  
Uses cosine annealing.

**Parameters**
- `optimizer`, `T_max`, `eta_min`, `last_epoch`.

**Output**  
Scheduler.

## `ReduceLROnPlateau`

**Syntax**
```python
ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
```

**What it does**  
Reduces learning rate when a monitored metric stops improving.

**Parameters**
- `optimizer`, `mode`, `factor`, `patience`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`, `eps`.

**Output**  
Scheduler.

## `OneCycleLR`

**Syntax**
```python
OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4, three_phase=False, last_epoch=-1)
```

**What it does**  
Uses a one-cycle learning-rate policy.

**Parameters**
- `optimizer`, `max_lr`, `total_steps`, `epochs`, `steps_per_epoch`, `pct_start`, `anneal_strategy`, momentum options, `div_factor`, `final_div_factor`, `three_phase`, `last_epoch`.

**Output**  
Scheduler.


# 18. torch.utils.data
## `Dataset`

**Syntax**
```python
class Dataset: __getitem__(index); __len__()
```

**What it does**  
Abstract map-style dataset interface.

**Parameters**
- Implementations define storage and indexed access.

**Output**  
Dataset object.

## `IterableDataset`

**Syntax**
```python
class IterableDataset
```

**What it does**  
Base class for iterable datasets.

**Parameters**
- Dataset-specific parameters.

**Output**  
Iterable dataset.

## `TensorDataset`

**Syntax**
```python
TensorDataset(*tensors)
```

**What it does**  
Wraps tensors as a dataset indexed along their first dimension.

**Parameters**
- `tensors` — tensors with matching first dimensions.

**Output**  
Dataset.

## `DataLoader`

**Syntax**
```python
DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='')
```

**What it does**  
Provides batching, sampling, multiprocessing and collation.

**Parameters**
- `dataset`, `batch_size`, `shuffle`, `sampler`, `batch_sampler`, `num_workers`, `collate_fn`, `pin_memory`, `drop_last`, `timeout`, `worker_init_fn`, `multiprocessing_context`, `generator`, `prefetch_factor`, `persistent_workers`, `pin_memory_device`.

**Output**  
DataLoader iterator.

## `random_split`

**Syntax**
```python
random_split(dataset, lengths, generator=None)
```

**What it does**  
Randomly partitions a dataset.

**Parameters**
- `dataset` — source.
- `lengths` — sizes or fractions.
- `generator` — RNG.

**Output**  
List of `Subset` objects.

## `Subset`

**Syntax**
```python
Subset(dataset, indices)
```

**What it does**  
View of a dataset restricted to selected indices.

**Parameters**
- `dataset` — source.
- `indices` — selected indices.

**Output**  
Subset.

## `ConcatDataset`

**Syntax**
```python
ConcatDataset(datasets)
```

**What it does**  
Concatenates map-style datasets.

**Parameters**
- `datasets` — sequence of datasets.

**Output**  
Combined dataset.

## `WeightedRandomSampler`

**Syntax**
```python
WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
```

**What it does**  
Samples indices according to weights.

**Parameters**
- `weights` — sampling weights.
- `num_samples` — number sampled.
- `replacement` — sample with replacement.
- `generator` — RNG.

**Output**  
Sampler.


# 19. torch.autograd
## `torch.autograd.backward`

**Syntax**
```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None)
```

**What it does**  
Computes gradients and accumulates them in leaf `.grad` fields.

**Parameters**
- `tensors` — outputs.
- `grad_tensors` — vector-Jacobian weights.
- `retain_graph` — retain graph.
- `create_graph` — higher-order graph.
- `inputs` — optional gradient targets.
- `grad_variables` — legacy.

**Output**  
`None`.

## `torch.autograd.grad`

**Syntax**
```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)
```

**What it does**  
Computes and returns gradients without accumulating them in `.grad`.

**Parameters**
- `outputs`, `inputs`, `grad_outputs`, `retain_graph`, `create_graph`, `allow_unused`, `is_grads_batched`, `materialize_grads`.
- `only_inputs` is deprecated/ignored.

**Output**  
Tuple or dictionary of gradients.

## `torch.no_grad`

**Syntax**
```python
torch.no_grad()
```

**What it does**  
Disables gradient calculation in a context or decorator.

**Parameters**
None.

**Output**  
Context manager/decorator.

## `torch.inference_mode`

**Syntax**
```python
torch.inference_mode(mode=True)
```

**What it does**  
Disables autograd and additional autograd overhead for inference.

**Parameters**
- `mode` — enable/disable.

**Output**  
Context manager/decorator.


# 20. CUDA, Devices and AMP
## `torch.device`

**Syntax**
```python
torch.device(type, index=None)
```

**What it does**  
Represents a compute device.

**Parameters**
- `type` — e.g. `cpu`, `cuda`, `mps`, `xpu`.
- `index` — optional device index.

**Output**  
`torch.device`.

## `torch.cuda.is_available`

**Syntax**
```python
torch.cuda.is_available()
```

**What it does**  
Checks CUDA availability.

**Parameters**
None.

**Output**  
Boolean.

## `torch.cuda.device_count`

**Syntax**
```python
torch.cuda.device_count()
```

**What it does**  
Returns visible CUDA device count.

**Parameters**
None.

**Output**  
Integer.

## `torch.cuda.get_device_name`

**Syntax**
```python
torch.cuda.get_device_name(device=None)
```

**What it does**  
Returns CUDA device name.

**Parameters**
- `device` — device index/object.

**Output**  
String.

## `torch.cuda.current_device`

**Syntax**
```python
torch.cuda.current_device()
```

**What it does**  
Returns current CUDA device index.

**Parameters**
None.

**Output**  
Integer.

## `torch.cuda.empty_cache`

**Syntax**
```python
torch.cuda.empty_cache()
```

**What it does**  
Releases unused cached CUDA memory.

**Parameters**
None.

**Output**  
`None`.

## `torch.autocast`

**Syntax**
```python
torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)
```

**What it does**  
Enables automatic mixed-precision execution for eligible operations.

**Parameters**
- `device_type` — accelerator type.
- `dtype` — autocast dtype.
- `enabled` — enable autocast.
- `cache_enabled` — weight cache.

**Output**  
Context manager/decorator.

## `torch.amp.GradScaler`

**Syntax**
```python
torch.amp.GradScaler(device='cuda', init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
```

**What it does**  
Scales gradients for reduced-precision training.

**Parameters**
- `device` — device type.
- `init_scale` — initial scale.
- `growth_factor` — scale growth.
- `backoff_factor` — scale reduction.
- `growth_interval` — successful iterations before growth.
- `enabled` — enable scaler.

**Output**  
GradScaler object.


# 21. Serialization and Model State
## `torch.save`

**Syntax**
```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True, _disable_byteorder_record=False)
```

**What it does**  
Serializes a Python object.

**Parameters**
- `obj` — object.
- `f` — path/file.
- `pickle_module` — serializer.
- `pickle_protocol` — protocol.
- serialization-format options.

**Output**  
`None`; writes serialized data.

## `torch.load`

**Syntax**
```python
torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)
```

**What it does**  
Loads serialized PyTorch data.

**Parameters**
- `f` — source.
- `map_location` — device remapping.
- `pickle_module` — loader.
- `weights_only` — restricted loading mode.
- `mmap` — memory-map storage.
- additional pickle arguments.

**Output**  
Deserialized object.

## `Module.state_dict`

**Syntax**
```python
module.state_dict(*, destination=None, prefix='', keep_vars=False)
```

**What it does**  
Returns module parameters and persistent buffers.

**Parameters**
- `destination` — mapping.
- `prefix` — key prefix.
- `keep_vars` — preserve variables.

**Output**  
Ordered mapping.

## `Module.load_state_dict`

**Syntax**
```python
module.load_state_dict(state_dict, strict=True, assign=False)
```

**What it does**  
Loads parameters and buffers.

**Parameters**
- `state_dict` — saved mapping.
- `strict` — exact key matching.
- `assign` — assignment behavior.

**Output**  
Named tuple with missing/unexpected keys.


# 22. Compilation and Performance
## `torch.compile`

**Syntax**
```python
torch.compile(model=None, *, fullgraph=False, dynamic=None, backend='inductor', mode=None, options=None, disable=False)
```

**What it does**  
Compiles a function or module for optimized execution.

**Parameters**
- `model` — callable/module.
- `fullgraph` — require full graph.
- `dynamic` — dynamic-shape behavior.
- `backend` — compiler backend.
- `mode` — optimization mode.
- `options` — backend options.
- `disable` — disable compilation.

**Output**  
Compiled callable/module.

## `Module.compile`

**Syntax**
```python
module.compile(*args, **kwargs)
```

**What it does**  
Compiles a module in place.

**Parameters**
- Compilation configuration arguments.

**Output**  
Compiled module.


# 23. Gradient Utilities
## `clip_grad_norm_`

**Syntax**
```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None)
```

**What it does**  
Clips gradients by their global norm.

**Parameters**
- `parameters` — parameter iterable.
- `max_norm` — maximum norm.
- `norm_type` — norm order.
- `error_if_nonfinite` — error on non-finite norm.
- `foreach` — implementation selection.

**Output**  
Total norm before clipping.

## `clip_grad_value_`

**Syntax**
```python
torch.nn.utils.clip_grad_value_(parameters, clip_value, foreach=None)
```

**What it does**  
Clips individual gradient values.

**Parameters**
- `parameters` — parameters.
- `clip_value` — absolute limit.
- `foreach` — implementation selection.

**Output**  
`None`.


# 24. RNN Utilities
## `pad_sequence`

**Syntax**
```python
pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side='right')
```

**What it does**  
Pads variable-length tensors.

**Parameters**
- `sequences` — tensors.
- `batch_first` — layout.
- `padding_value` — fill value.
- `padding_side` — left/right.

**Output**  
Padded tensor.

## `pack_padded_sequence`

**Syntax**
```python
pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
```

**What it does**  
Packs padded variable-length sequences.

**Parameters**
- `input` — padded sequence.
- `lengths` — lengths.
- `batch_first` — layout.
- `enforce_sorted` — sorting requirement.

**Output**  
`PackedSequence`.

## `pad_packed_sequence`

**Syntax**
```python
pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)
```

**What it does**  
Unpacks a packed sequence.

**Parameters**
- `sequence` — packed sequence.
- `batch_first` — output layout.
- `padding_value` — fill value.
- `total_length` — optional total length.

**Output**  
Tuple `(padded_output, lengths)`.


# 25. Linear Algebra and FFT
## `torch.linalg.norm`

**Syntax**
```python
torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None, out=None)
```

**What it does**  
Computes vector or matrix norms.

**Parameters**
- `A` — tensor.
- `ord` — norm order.
- `dim` — dimensions.
- `keepdim` — retain dimensions.
- `dtype` — computation dtype.
- `out` — destination.

**Output**  
Norm tensor.

## `torch.linalg.solve`

**Syntax**
```python
torch.linalg.solve(A, B, *, left=True)
```

**What it does**  
Solves linear systems.

**Parameters**
- `A` — coefficient matrix.
- `B` — right side.
- `left` — solve orientation.

**Output**  
Solution tensor.

## `torch.linalg.inv`

**Syntax**
```python
torch.linalg.inv(A, *, out=None)
```

**What it does**  
Computes matrix inverse.

**Parameters**
- `A` — square matrix.
- `out` — destination.

**Output**  
Inverse matrix.

## `torch.linalg.det`

**Syntax**
```python
torch.linalg.det(A)
```

**What it does**  
Computes determinant.

**Parameters**
- `A` — square matrix or batch.

**Output**  
Determinant tensor.

## `torch.linalg.eig`

**Syntax**
```python
torch.linalg.eig(A)
```

**What it does**  
Computes eigenvalues and eigenvectors.

**Parameters**
- `A` — square matrix/batch.

**Output**  
Tuple `(eigenvalues, eigenvectors)`.

## `torch.linalg.svd`

**Syntax**
```python
torch.linalg.svd(A, full_matrices=True, *, driver=None)
```

**What it does**  
Computes singular value decomposition.

**Parameters**
- `A` — matrix/batch.
- `full_matrices` — full/reduced decomposition.
- `driver` — backend algorithm.

**Output**  
Tuple `(U, S, Vh)`.

## `torch.fft.fft`

**Syntax**
```python
torch.fft.fft(input, n=None, dim=-1, norm=None)
```

**What it does**  
Computes 1-D discrete Fourier transform.

**Parameters**
- `input` — tensor.
- `n` — transform size.
- `dim` — transform dimension.
- `norm` — normalization.

**Output**  
Complex tensor.

## `torch.fft.fft2`

**Syntax**
```python
torch.fft.fft2(input, s=None, dim=(-2,-1), norm=None)
```

**What it does**  
Computes 2-D discrete Fourier transform.

**Parameters**
- `input` — tensor.
- `s` — transform sizes.
- `dim` — transform dimensions.
- `norm` — normalization.

**Output**  
Complex tensor.

## `torch.fft.ifft`

**Syntax**
```python
torch.fft.ifft(input, n=None, dim=-1, norm=None)
```

**What it does**  
Computes inverse 1-D Fourier transform.

**Parameters**
- `input`, `n`, `dim`, `norm`.

**Output**  
Complex tensor.


# 26. Reproducibility and Randomness
## `torch.manual_seed`

**Syntax**
```python
torch.manual_seed(seed)
```

**What it does**  
Sets PyTorch's random seed.

**Parameters**
- `seed` — integer seed.

**Output**  
`torch.Generator`.

## `torch.use_deterministic_algorithms`

**Syntax**
```python
torch.use_deterministic_algorithms(mode, *, warn_only=False)
```

**What it does**  
Requests deterministic algorithms where available.

**Parameters**
- `mode` — enable/disable.
- `warn_only` — warn instead of error when unavailable.

**Output**  
`None`.


# 27. torchvision — Transforms and Models
## `transforms.Compose`

**Syntax**
```python
transforms.Compose(transforms)
```

**What it does**  
Chains transforms sequentially.

**Parameters**
- `transforms` — transformation sequence.

**Output**  
Callable transform pipeline.

## `transforms.Resize`

**Syntax**
```python
transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
```

**What it does**  
Resizes images.

**Parameters**
- `size` — target size.
- `interpolation` — interpolation mode.
- `max_size` — optional maximum.
- `antialias` — antialiasing.

**Output**  
Resized image.

## `transforms.CenterCrop`

**Syntax**
```python
transforms.CenterCrop(size)
```

**What it does**  
Crops the center region.

**Parameters**
- `size` — crop dimensions.

**Output**  
Cropped image.

## `transforms.RandomCrop`

**Syntax**
```python
transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
```

**What it does**  
Randomly crops an image.

**Parameters**
- `size` — crop size.
- `padding` — padding.
- `pad_if_needed` — pad if smaller.
- `fill` — fill value.
- `padding_mode` — padding mode.

**Output**  
Cropped image.

## `transforms.RandomHorizontalFlip`

**Syntax**
```python
transforms.RandomHorizontalFlip(p=0.5)
```

**What it does**  
Randomly flips horizontally.

**Parameters**
- `p` — probability.

**Output**  
Image.

## `transforms.RandomRotation`

**Syntax**
```python
transforms.RandomRotation(degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0)
```

**What it does**  
Randomly rotates images.

**Parameters**
- `degrees` — rotation range.
- `interpolation` — interpolation.
- `expand` — expand bounds.
- `center` — center.
- `fill` — fill value.

**Output**  
Rotated image.

## `transforms.ToTensor`

**Syntax**
```python
transforms.ToTensor()
```

**What it does**  
Converts supported images/arrays to tensors with supported image scaling.

**Parameters**
None.

**Output**  
Tensor image.

## `transforms.Normalize`

**Syntax**
```python
transforms.Normalize(mean, std, inplace=False)
```

**What it does**  
Normalizes tensor image channels.

**Parameters**
- `mean` — channel means.
- `std` — channel standard deviations.
- `inplace` — in-place operation.

**Output**  
Normalized tensor.

## `torchvision.models.<model>`

**Syntax**
```python
torchvision.models.<model>(weights=None, ...)
```

**What it does**  
Constructs a vision architecture, optionally with pretrained weights.

**Parameters**
- `weights` — pretrained weight configuration or `None`.
- Additional parameters are architecture-specific.

**Output**  
`nn.Module`.


# 28. Module Control and Training
## `Module.train`

**Syntax**
```python
module.train(mode=True)
```

**What it does**  
Sets module and children to training mode.

**Parameters**
- `mode` — training flag.

**Output**  
Module itself.

## `Module.eval`

**Syntax**
```python
module.eval()
```

**What it does**  
Sets module and children to evaluation mode.

**Parameters**
None.

**Output**  
Module itself.

## `Module.parameters`

**Syntax**
```python
module.parameters(recurse=True)
```

**What it does**  
Iterates over parameters.

**Parameters**
- `recurse` — include submodules.

**Output**  
Iterator of `Parameter`.

## `Module.named_parameters`

**Syntax**
```python
module.named_parameters(prefix='', recurse=True, remove_duplicate=True)
```

**What it does**  
Iterates over parameter names and parameters.

**Parameters**
- `prefix` — name prefix.
- `recurse` — include submodules.
- `remove_duplicate` — remove duplicates.

**Output**  
Iterator of `(name, parameter)`.

## `Module.named_buffers`

**Syntax**
```python
module.named_buffers(prefix='', recurse=True, remove_duplicate=True)
```

**What it does**  
Iterates over buffer names and buffers.

**Parameters**
- `prefix` — prefix.
- `recurse` — include submodules.
- `remove_duplicate` — remove duplicates.

**Output**  
Iterator of `(name, buffer)`.


# Quick Selection Map

| Task | Primary API |
|---|---|
| Tensor creation | `torch.tensor`, `zeros`, `ones`, `rand`, `randn`, `randint` |
| Shape changes | `reshape`, `view`, `flatten`, `squeeze`, `unsqueeze`, `permute` |
| Tensor combination | `cat`, `stack`, `chunk`, `split` |
| Neural networks | `torch.nn` |
| Functional operations | `torch.nn.functional` |
| Initialization | `torch.nn.init` |
| Losses | `torch.nn` / `torch.nn.functional` |
| Optimization | `torch.optim` |
| LR schedules | `torch.optim.lr_scheduler` |
| Datasets | `torch.utils.data.Dataset` |
| Batching | `torch.utils.data.DataLoader` |
| Automatic differentiation | `torch.autograd` |
| GPU | `torch.cuda` / `torch.device` |
| Mixed precision | `torch.amp` |
| Saving/loading | `torch.save`, `torch.load`, `state_dict` |
| Compilation | `torch.compile` |
| Distributed training | `torch.distributed` |
| Computer vision | `torchvision` |
| Linear algebra | `torch.linalg` |
| Fourier transforms | `torch.fft` |

## Official Documentation

- PyTorch: https://docs.pytorch.org/docs/stable/
- Tutorials: https://docs.pytorch.org/tutorials/
- Torchvision: https://docs.pytorch.org/vision/stable/
