from typing import Union, Callable, Optional
import sys
import torch

from flashrnn import FlashRNNConfig, flashrnn
from flashrnn.flashrnn.flashrnn import _zero_state


def torch_dtype_to_str(dtype: torch.dtype):
    if dtype == torch.float:
        return "float32"
    else:
        return str(dtype)[6:]


def plot_diff(x: torch.Tensor, y: torch.Tensor, **kwargs):
    import matplotlib.pyplot as plt

    if "index" in kwargs:
        index = kwargs["index"]
        kwargs.pop("index")
    else:
        index = 0

    fig, ax = plt.subplots()
    for s in x.shape:
        if s != 1:
            x1 = s
            break
    x = x.reshape(x1, -1)
    y = y.reshape(x1, -1)
    im = ax.imshow((x - y).abs().cpu().detach(), **kwargs)
    fig.colorbar(im, ax=ax)

    fig.savefig(f"errorplot_{index}.png")


def output_shape(x: Union[torch.Tensor, tuple[torch.Tensor]]) -> str:
    if isinstance(x, tuple) or isinstance(x, list):
        return "[" + ", ".join(str(xi.shape) for xi in x) + "]"
    else:
        str(x.shape)


def check_forward(
    func1,
    func2,
    inputs: Union[torch.Tensor, tuple[torch.Tensor]],
    verbose=True,
    show_plot_diff: bool = True,
    plot_diff_kwargs: dict = {},
    tensor_compare: Callable[[torch.Tensor, torch.Tensor], bool] = torch.allclose,
    **tensor_compare_kwargs,
) -> bool:
    if isinstance(inputs, torch.Tensor):
        res1 = func1(inputs)
    else:
        res1 = func1(*inputs)
    if isinstance(inputs, torch.Tensor):
        res2 = func2(inputs)
    else:
        res2 = func2(*inputs)

    if isinstance(res1, tuple) or isinstance(res2, list):
        if len(res1) != len(res2) or not (
            isinstance(res2, tuple) or isinstance(res2, list)
        ):
            if verbose:
                print(
                    f"Invalid output vars: {output_shape(res1)} != {output_shape(res2)}",
                    file=sys.stderr,
                )
            return False
        same = True
        for i, _ in enumerate(res1):
            if res1[i].shape != res2[i].shape:
                if verbose:
                    print(
                        f"Shape mismatch {i}: {res1[i].shape} != {res2[i].shape}",
                        file=sys.stderr,
                    )
                same = False
            if not tensor_compare(res1[i], res2[i], **tensor_compare_kwargs):
                if show_plot_diff:
                    plot_diff(res1[i], res2[i], **plot_diff_kwargs)
                if verbose:
                    print(f"Value mismatch {i}: ", file=sys.stderr)
                same = False
        return same
    else:
        if res1.shape != res2.shape:
            if verbose:
                print(
                    f"Forward invalid output shape: {output_shape(res1)} != {output_shape(res2)}",
                    file=sys.stderr,
                )
            return False
        if not tensor_compare(res1, res2, **tensor_compare_kwargs):
            if show_plot_diff:
                plot_diff(res1, res2, **plot_diff_kwargs)
            if verbose:
                print("Forward value mismatch.", file=sys.stderr)
            return False
        return True


def check_backward(
    func1,
    func2,
    inputs: Union[torch.Tensor, tuple[torch.Tensor]],
    input_grad_mask: Optional[
        Union[
            list[Optional[Callable[[torch.Tensor], torch.Tensor]]],
            Callable[[torch.Tensor], torch.Tensor],
        ]
    ] = None,
    verbose=True,
    show_plot_diff=True,
    tensor_compare: Callable[[torch.Tensor, torch.Tensor], bool] = torch.allclose,
    plot_diff_kwargs: dict = {},
    **tensor_compare_kwargs,
) -> bool:
    if isinstance(inputs, torch.Tensor):
        inputs1 = inputs.clone().detach()
        inputs2 = inputs.clone().detach()
        inputs2.requires_grad_(True)
        inputs1.requires_grad_(True)
        res1 = func1(inputs1)
        res2 = func2(inputs2)
    else:
        inputs1 = [inp.clone().detach() if inp is not None else None for inp in inputs]
        inputs2 = [inp.clone().detach() if inp is not None else None for inp in inputs]
        for inp in inputs2:
            if inp is not None:
                inp.requires_grad_(True)
        for inp in inputs1:
            if inp is not None:
                inp.requires_grad_(True)
        res1 = func1(*inputs1)
        res2 = func2(*inputs2)

    # print(res1.grad_fn, inputs[0].requires_grad)
    if isinstance(res1, torch.Tensor):
        masks = torch.randn_like(res1)
        masks2 = masks.clone().detach()
        (masks2 * res2).sum().backward()
        (masks * res1).sum().backward()
    else:
        masks = [torch.randn_like(y1) for y1 in res1]
        masks2 = [y1.clone().detach() for y1 in masks]
        sum((m1 * y1).sum() for m1, y1 in zip(masks, res1)).backward()
        sum((m2 * y2).sum() for m2, y2 in zip(masks2, res2)).backward()

    same = True
    if isinstance(inputs1, torch.Tensor):
        same = True
        if inputs1.grad is None:
            if verbose:
                print("No grad for func1", file=sys.stderr)
            same = False
        if inputs2.grad is None:
            if verbose:
                print("No grad for func2", file=sys.stderr)
            same = False
        if input_grad_mask is not None:
            ig1 = input_grad_mask(inputs1.grad)
            ig2 = input_grad_mask(inputs2.grad)
        else:
            ig1 = inputs1.grad
            ig2 = inputs2.grad
        if not tensor_compare(ig1, ig2, **tensor_compare_kwargs):
            if verbose:
                print("Backward value mismatch", file=sys.stderr)
            if show_plot_diff:
                plot_diff(ig1, ig2, **plot_diff_kwargs)
            same = False
    else:
        for i, _ in enumerate(inputs1):
            if inputs1[i] is None:
                continue
            if inputs1[i].grad is None and inputs2[i].grad is None:
                continue
            if inputs1[i].grad is None:
                if verbose:
                    print(f"No grad for func1 {i}", file=sys.stderr)
                same = False
            if inputs2[i].grad is None:
                if verbose:
                    print(f"No grad for func2 {i}", file=sys.stderr)
                same = False
            if input_grad_mask is not None and input_grad_mask[i] is not None:
                ig1 = input_grad_mask[i](inputs1[i].grad)
                ig2 = input_grad_mask[i](inputs2[i].grad)
            else:
                ig1 = inputs1[i].grad
                ig2 = inputs2[i].grad
            if not tensor_compare(ig1, ig2, **tensor_compare_kwargs):
                if show_plot_diff:
                    plot_diff(ig1, ig2, index=i, **plot_diff_kwargs)
                if verbose:
                    print(
                        f"Backward value mismatch {i}: max_abs_diff: {(inputs1[i].grad- inputs2[i].grad).abs().max()}",
                        file=sys.stderr,
                    )
                same = False

    return same


def create_inputs(
    batch_size: int,
    sequence_size: int,
    num_heads: int,
    head_dim: int,
    function: str,
    create_states: bool = True,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    **kwargs,
):
    cfg = FlashRNNConfig(
        batch_size=batch_size,
        num_heads=num_heads,
        function=function,
        head_dim=head_dim,
        dtype=torch_dtype_to_str(dtype),
    )

    num_gates_w = cfg.num_gates_w
    num_gates_r = cfg.num_gates_r
    num_gates_t = cfg.num_gates_t

    Wx = torch.randn(
        [batch_size, sequence_size, num_gates_w, num_heads, head_dim],
        device=device,
        dtype=dtype,
    )
    R = torch.randn(
        [num_gates_r, num_heads, head_dim, head_dim],
        device=device,
        dtype=dtype,
    ) / head_dim ** (0.5)
    b = torch.randn(
        [num_gates_t, num_heads, head_dim],
        device=device,
        dtype=dtype,
    )
    states = _zero_state(cfg, Wx)
    assert states.dtype == dtype

    if create_states:
        return Wx, states, R, b
    else:
        return Wx, R, b


def model_test(
    batch_size: int,
    sequence_size: int,
    num_heads: int,
    head_dim: int,
    function: str,
    backend="cuda_fused",
    backend_cmp="vanilla",
    device="cuda",
    dtype=torch.float16,
    include_backward: bool = True,
    tensor_compare_kwargs: dict = {},
    show_plot_diff: bool = False,
):
    Wx, states, R, b = create_inputs(
        batch_size,
        sequence_size=sequence_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        create_states=True,
        device=device,
        function=function,
    )

    res = check_forward(
        lambda Wx, states, R, b: flashrnn(
            Wx,
            states,
            R,
            b,
            function,
            backend=backend,
            dtype=torch_dtype_to_str(dtype),
        ),
        lambda Wx, states, R, b: flashrnn(
            Wx,
            states,
            R,
            b,
            function,
            backend=backend_cmp,
            dtype=torch_dtype_to_str(dtype),
        ),
        inputs=(Wx, R, b, states),
        show_plot_diff=show_plot_diff,
        **tensor_compare_kwargs,
    )

    if include_backward:
        res = (
            res
            and check_backward(
                lambda Wx, R, b, states: flashrnn(
                    Wx,
                    R,
                    b,
                    states,
                    function=function,
                    backend=backend,
                    dtype=torch_dtype_to_str(dtype),
                )[0][0],
                lambda Wx, R, b, states: flashrnn(
                    Wx,
                    R,
                    b,
                    states,
                    function=function,
                    backend=backend_cmp,
                    dtype=torch_dtype_to_str(dtype),
                )[0][0],
                input_grad_mask=[
                    None,
                    None,
                    None,
                    lambda x: torch.tensor(0.0),
                ],  # do no test initial state gradients for empty default state (different m behavior)
                inputs=(Wx, R, b, states),
                show_plot_diff=False,
                **tensor_compare_kwargs,
            )
        )

    return res
