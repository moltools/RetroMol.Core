"""
Author:         David Meijer
Licence:        MIT License
Description:    Error parsing functions.

Code adapted from David Vujic's implementation of railway oriented programming
in Python: https://github.com/DavidVujic/pythonic-railway
"""
import traceback 
import typing as ty
from dataclasses import dataclass 
from functools import partial 

@dataclass 
class Fail:
    """
    Dataclass for storing information about a failed function call.

    Attributes
    ----------
    exception : Exception | None, optional
        Exception that was raised, by default None.
    stacktrace : str, optional
        Stacktrace of the exception, by default None.
    name : str | None, optional
        Name of the function that was called, by default None.
    args : ty.Any, optional
        Positional arguments that were passed to the function, by default None.
    kwargs : ty.Any, optional
        Keyword arguments that were passed to the function, by default None.
    """
    exception: Exception | None = None 
    stacktrace: str = None 
    name: str | None = None 
    args: ty.Any = None 
    kwargs: ty.Any = None 

@dataclass 
class Success:
    """
    Dataclass for storing information about a successful function call.

    Attributes
    ----------
    value : ty.Any, optional
        Return value of the function, by default None.
    """
    value: ty.Any = None 

def failed(result: ty.Union[Success, Fail]) -> bool:
    """
    Checks whether the given result is a Fail instance.

    Parameters
    ----------
    result : ty.Union[Success, Fail]
        Result to check.
    
    Returns
    -------
    bool
        Whether the given result is a Fail instance.
    """
    return isinstance(result, Fail)

def succeeded(result: ty.Union[Success, Fail]) -> bool:
    """
    Checks whether the given result is a Success instance.
    
    Parameters
    ----------
    result : ty.Union[Success, Fail]
        Result to check.
    
    Returns
    -------
    bool
        Whether the given result is a Success instance.
    """
    return isinstance(result, Success)

def try_catch(func: ty.Callable, *args, **kwargs) -> ty.Union[Success, Fail]:
    """
    Tries to call the given function with the given arguments and returns a
    
    Parameters
    ----------
    func : ty.Callable
        Function to call.
    args : ty.Any
        Positional arguments to pass to the function.
    kwargs : ty.Any
        Keyword arguments to pass to the function.
    
    Returns
    -------
    ty.Union[Success, Fail]
        Success instance if the function call was successful, Fail instance
        otherwise.
    """
    try: 
        return Success(func(*args, **kwargs))
    
    except Exception as exc:
        return Fail(exc, traceback.format_exc(), func.__name__, args, kwargs)
    
def evaluation_wrapper(
    error_handling: ty.Callable, 
    func: ty.Callable, 
    *args, 
    **kwargs,
) -> ty.Any:
    """
    Evaluates the given function with the given arguments and returns the result
    of the function call if it was successful, otherwise it returns the result of
    the error handling function.
    
    Parameters
    ----------
    error_handling : ty.Callable
        Error handling function to call if the function call was unsuccessful.
    func : ty.Callable
        Function to call.
    args : ty.Any
        Positional arguments to pass to the function.
    kwargs : ty.Any
        Keyword arguments to pass to the function.
    
    Returns
    -------
    ty.Any
        Result of the function call if it was successful, otherwise the result of
        the error handling function.
    """
    match args:
        # If the function call was successful, return the result of the function
        # call.
        case [result, *_] if len(args) and succeeded(result):
            return result.value 
        
        # If the function call was unsuccessful, return the result of the error
        # handling function.
        case _:
            return error_handling(func, *args, **kwargs)

def error_handler(func: ty.Callable) -> ty.Callable:
    """
    Decorator for wrapping a function call in a try-catch block and returning a
    Fail instance if the function call was unsuccessful.
    
    Parameters
    ----------
    func : ty.Callable
        Function to wrap.
    
    Returns
    -------
    ty.Callable
        Wrapped function.
    """
    return partial(evaluation_wrapper, try_catch, func)

def resolve(result: ty.Union[Success, Fail]) -> ty.Any:
    """
    Resolves the given result by returning the value of the result if it was a
    Success instance, otherwise it raises the exception of the result.
    
    Parameters
    ----------
    result : ty.Union[Success, Fail]
        Result to resolve.
    
    Returns
    -------
    ty.Any
        Value of the result if it was a Success instance.
    
    Raises
    ------
    Exception
        Exception of the result if it was a Fail instance.
    """
    match result:
        # If the result was a Success instance, return the value of the result.
        case result if succeeded(result):
            return result.value 
        
        # If the result was a Fail instance, raise the exception of the result.
        case fail:
            raise fail.exception 
