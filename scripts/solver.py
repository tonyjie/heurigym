import time
import os

def over_time(start_time, time_limit):
    return (time.time() - start_time) > time_limit

def solve(input_file: str, solution_file: str):
    """
    Solve the optimization problem.

    Please do NOT change the function name and arguments.
    Inputs should be read from input_file and outputs should be written to solution_file.
    Input and output formats have been specified in the problem statement.
    MUST honor the time_limit constraint. Use the over_time(start_time, time_limit) function in every major loop.
    """
    time_limit = int(os.getenv("SOLVER_TIMEOUT", 120))
    start_time = time.time()

    # Example of how timeout should be checked inside loops:
    for t in range(1):  # placeholder loop
        if over_time(start_time, time_limit):
            break


    raise NotImplementedError(
        "This is a placeholder implementation you need to fill in."
    )
