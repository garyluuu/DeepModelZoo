from typing import Callable
import importlib
import deepMZ



inputModes = ["mvWin", "allin", "last"]

__seqTools = importlib.import_module(f".SeqWrap{deepMZ.backend}", "deepMZ.temporal")

nn = importlib.import_module(f".net{deepMZ.backend}", "deepMZ.temporal")


def wrapper(
    net,
    # evo_net,
    evolveLen: int = 1,
    inputMode: str | Callable = "last",
    inputFnArgs = {},
    hasAuxiliary=False,
    outAuxiliary=False,
    **kwargs,
) -> Callable:
    """
    wrapper function for next-step predictor.
    * net: the network to be evolved.
    * evolveLen: the number of time iteration.
    * inputMode: the input mode, can be one of "mvWin", "all", "last", or a callable function.
    """

    assert (
        inputMode in inputModes
    ), f"inputMode must be one of {inputModes}, but got {inputMode}"

    if isinstance(inputMode, str):
        inputFn = getattr(__seqTools, inputMode)(**inputFnArgs)
    else: inputFn = inputMode
    
    # return __seqTools.SeqWrapper(net, evo_net, evolveLen, inputFn, hasAuxiliary, outAuxiliary,**kwargs)
    return __seqTools.SeqWrapper(net, evolveLen, inputFn, hasAuxiliary, outAuxiliary,**kwargs)

# def wrapper_test(
#     net,
#     testLen: int = 1,
#     inputMode: str | Callable = "last",
#     inputFnArgs = {},
#     hasAuxiliary=False,
#     outAuxiliary=False,
#     **kwargs,
# ) -> Callable:
#     """
#     wrapper function for next-step predictor.
#     * net: the network to be evolved.
#     * inputMode: the input mode, can be one of "mvWin", "all", "last", or a callable function.
#     """

#     assert (
#         inputMode in inputModes
#     ), f"inputMode must be one of {inputModes}, but got {inputMode}"

#     if isinstance(inputMode, str):
#         inputFn = getattr(__seqTools, inputMode)(**inputFnArgs)
#     else: inputFn = inputMode
    
#     return __seqTools.SeqWrapper(net, testLen, inputFn, hasAuxiliary, outAuxiliary,**kwargs)