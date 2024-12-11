import logging
from typing import Any, Callable, Sequence

from ..autotune import constrint as cst
from .cuda_init import defines_to_cflags, load

LOGGER = logging.getLogger(__name__)


def load_parametric(
    *,
    name: str,
    sources,
    constraint_str: str,
    value_refinements: Sequence[cst.ValueRefinement],
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    **kwargs,
):
    constr, var = cst.parse_expression(constraint_str)
    sols = cst.solve_constrint(constr, var, value_refinements, 1)

    if not sols:
        raise ValueError("No solution of constraints for CUDA module")

    LOGGER.info("Using kernel parameters:", sols[0])

    extra_cflags = list(extra_cflags) + list(defines_to_cflags(sols[0]))

    return load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        **kwargs,
    )


def load_parametric_and_test(
    *,
    name: str,
    sources,
    constraint_str: str,
    value_refinements: Sequence[cst.ValueRefinement],
    model_class: str,
    model_args: tuple,
    test_input: Any,
    map_output_to_backward_input: Callable,
    max_trials: int = 25,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    **kwargs,
):
    constr, var = cst.parse_expression(constraint_str)
    excluded_solutions = []
    error_on_test = True
    trials = 0
    while error_on_test:
        sols = cst.solve_constrint(
            constr, var, value_refinements, 1, exclude_solutions=excluded_solutions
        )

        if not sols or trials > max_trials:
            raise ValueError("No working solution of constraints for CUDA module")

        LOGGER.info(f"Trying kernel parameters: {sols[0]}")

        cflags = list(extra_cflags) + list(defines_to_cflags(sols[0]))

        module = load(
            name=name,
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            **kwargs,
        )
        model = getattr(module, model_class)(*model_args)

        error_on_test = False
        try:
            res = model.forward(*test_input)
            _ = model.backward(*map_output_to_backward_input(test_input, res))
        except RuntimeError as err:
            LOGGER.info(f"Failed using kernel: {err}")

            excluded_solutions.append(sols[0])
            error_on_test = True
            trials += 1
    return module


def load_parametric_and_test_and_bisect(
    *,
    name: str,
    sources,
    constraint_str: str,
    value_refinements: Sequence[cst.ValueRefinement],
    model_class: str,
    model_args: tuple,
    test_input: Any,
    map_output_to_backward_input: Callable,
    value_to_independently_bisect_upwards_forward: str,
    value_to_independently_bisect_upwards_backward: str,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    check_backward: bool = True,
    **kwargs,
):
    constr, var = cst.parse_expression(constraint_str)
    excluded_solutions = []
    error_on_test = True
    trials = 0

    # TODO: make this work for multiple paramters
    forward_min_value = {}
    forward_max_value = {}
    backward_min_value = {}
    backward_max_value = {}
    forward_min_value[value_to_independently_bisect_upwards_forward] = 1
    forward_max_value[value_to_independently_bisect_upwards_forward] = None
    backward_min_value[value_to_independently_bisect_upwards_backward] = 1
    backward_max_value[value_to_independently_bisect_upwards_backward] = None

    while error_on_test:
        found_forward = False
        found_backward = not check_backward
        res = None

        while not (found_forward and found_backward):
            sols = cst.solve_constrint(
                constr, var, value_refinements, 1, exclude_solutions=excluded_solutions
            )

            if not sols:
                LOGGER.info(
                    f"No working solution of constraints for CUDA module: {constr}"
                )
                raise ValueError("No working solution of constraints for CUDA module")

            LOGGER.info(f"Trying kernel parameters: {sols[0]}")

            cflags = list(extra_cflags) + list(defines_to_cflags(sols[0]))

            module = load(
                name=name,
                sources=sources,
                extra_cflags=cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                **kwargs,
            )
            model = getattr(module, model_class)(*model_args)
            try:
                res = model.forward(*test_input)
                if (
                    forward_max_value[value_to_independently_bisect_upwards_forward]
                    is None
                    or forward_max_value[value_to_independently_bisect_upwards_forward]
                    - forward_min_value[value_to_independently_bisect_upwards_forward]
                    <= 1
                ):
                    found_forward = True
                    var[value_to_independently_bisect_upwards_forward] = (
                        cst.IntegerVariable(
                            values=(
                                sols[0][value_to_independently_bisect_upwards_forward],
                            )
                        )
                    )
                else:
                    forward_min_value[value_to_independently_bisect_upwards_forward] = (
                        sols[0][value_to_independently_bisect_upwards_forward]
                    )
                    var[value_to_independently_bisect_upwards_forward] = (
                        cst.IntegerVariable(
                            values=(
                                (
                                    1
                                    + forward_max_value[
                                        value_to_independently_bisect_upwards_forward
                                    ]
                                    + forward_min_value[
                                        value_to_independently_bisect_upwards_forward
                                    ]
                                )
                                // 2,
                            )
                        )
                    )
                    continue
            except RuntimeError as err:
                LOGGER.info(f"Failed using kernel: {err}")
                if (
                    forward_max_value[value_to_independently_bisect_upwards_forward]
                    is not None
                    and forward_max_value[value_to_independently_bisect_upwards_forward]
                    - forward_min_value[value_to_independently_bisect_upwards_forward]
                    <= 1
                ):
                    found_forward = True
                    var[value_to_independently_bisect_upwards_forward] = (
                        cst.IntegerVariable(
                            values=(
                                forward_min_value[
                                    value_to_independently_bisect_upwards_forward
                                ],
                            )
                        )
                    )
                else:
                    forward_max_value[value_to_independently_bisect_upwards_forward] = (
                        sols[0][value_to_independently_bisect_upwards_forward]
                    )
                    var[value_to_independently_bisect_upwards_forward] = (
                        cst.IntegerVariable(
                            values=(
                                (
                                    1
                                    + forward_max_value[
                                        value_to_independently_bisect_upwards_forward
                                    ]
                                    + forward_min_value[
                                        value_to_independently_bisect_upwards_forward
                                    ]
                                )
                                // 2,
                            )
                        )
                    )
                    continue

            if check_backward and res is not None:
                try:
                    _ = model.backward(*map_output_to_backward_input(test_input, res))
                    if (
                        backward_max_value[
                            value_to_independently_bisect_upwards_backward
                        ]
                        is None
                        or backward_max_value[
                            value_to_independently_bisect_upwards_backward
                        ]
                        - backward_min_value[
                            value_to_independently_bisect_upwards_backward
                        ]
                        <= 1
                    ):
                        found_backward = True
                        var[value_to_independently_bisect_upwards_backward] = (
                            cst.IntegerVariable(
                                values=(
                                    sols[0][
                                        value_to_independently_bisect_upwards_backward
                                    ],
                                )
                            )
                        )
                    else:
                        backward_min_value[
                            value_to_independently_bisect_upwards_backward
                        ] = sols[0][value_to_independently_bisect_upwards_backward]
                        var[value_to_independently_bisect_upwards_backward] = (
                            cst.IntegerVariable(
                                values=(
                                    (
                                        1
                                        + backward_max_value[
                                            value_to_independently_bisect_upwards_backward
                                        ]
                                        + backward_min_value[
                                            value_to_independently_bisect_upwards_backward
                                        ]
                                    )
                                    // 2,
                                )
                            )
                        )
                except RuntimeError as err:
                    LOGGER.info(f"Failed using kernel: {err}")
                    if (
                        backward_max_value[
                            value_to_independently_bisect_upwards_backward
                        ]
                        is not None
                        and backward_max_value[
                            value_to_independently_bisect_upwards_backward
                        ]
                        - backward_min_value[
                            value_to_independently_bisect_upwards_backward
                        ]
                        <= 1
                    ):
                        found_backward = True
                        var[value_to_independently_bisect_upwards_backward] = (
                            cst.IntegerVariable(
                                values=(
                                    backward_min_value[
                                        value_to_independently_bisect_upwards_backward
                                    ],
                                )
                            )
                        )
                    else:
                        backward_max_value[
                            value_to_independently_bisect_upwards_backward
                        ] = sols[0][value_to_independently_bisect_upwards_backward]
                        var[value_to_independently_bisect_upwards_backward] = (
                            cst.IntegerVariable(
                                values=(
                                    (
                                        1
                                        + backward_max_value[
                                            value_to_independently_bisect_upwards_backward
                                        ]
                                        + backward_min_value[
                                            value_to_independently_bisect_upwards_backward
                                        ]
                                    )
                                    // 2,
                                )
                            )
                        )
                        continue
            if res is None:
                raise RuntimeError("Failed using Kernel")
        sols = cst.solve_constrint(
            constr, var, value_refinements, 1, exclude_solutions=excluded_solutions
        )

        if not sols:
            LOGGER.info(f"No working solution of constraints for CUDA module: {constr}")
            raise ValueError("No working solution of constraints for CUDA module")

        LOGGER.info(f"Trying kernel parameters: {sols[0]}")

        cflags = list(extra_cflags) + list(defines_to_cflags(sols[0]))

        module = load(
            name=name,
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            **kwargs,
        )
        model = getattr(module, model_class)(*model_args)

        error_on_test = False
        try:
            res = model.forward(*test_input)
            if check_backward:
                _ = model.backward(*map_output_to_backward_input(test_input, res))
        except RuntimeError as err:
            LOGGER.info(f"Failed using kernel: {err}")

            excluded_solutions.append(sols[0])
            error_on_test = True
            trials += 1

    return module
