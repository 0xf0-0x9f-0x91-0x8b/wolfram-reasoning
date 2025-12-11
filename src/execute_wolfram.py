import subprocess
import pandas as pd

# ------------------------------------------------------
# Execute Wolfram code using wolframscript
# ------------------------------------------------------
def run_wolfram(code: str, timeout: int = 10):
    """
    Execute Wolfram / Mathematica code using wolframscript.
    
    Parameters:
        code (str): Wolfram Language code (string)
        timeout (int): seconds before killing the process
    
    Returns:
        (stdout, stderr, returncode)
    """
    cmd = ["wolframscript", "-code", code]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        return stdout, stderr, proc.returncode

    except subprocess.TimeoutExpired:
        return "", "Execution timed out", -1

    except Exception as e:
        return "", f"Python exception: {str(e)}", -1


# ------------------------------------------------------
# Apply Wolfram execution to an entire DataFrame column
# ------------------------------------------------------
def evaluate_wolfram_df(df: pd.DataFrame, col: str = "predicted"):
    """
    Given a DataFrame with a column containing Wolfram code,
    execute each row and store results into new columns.

    Output columns:
        wolfram_output
        wolfram_error
        wolfram_returncode
    """

    outputs = []
    errors = []
    codes = []

    for idx, code in df[col].items():
        if not isinstance(code, str):
            outputs.append("")
            errors.append("")
            codes.append("")
            continue
        stdout, stderr, ret = run_wolfram(code + "\nf")
        outputs.append(stdout)
        errors.append(stderr)
        codes.append(ret)

    df["wolfram_output"] = outputs
    df["wolfram_error"] = errors
    df["wolfram_returncode"] = codes

    return df


# ------------------------------------------------------
# Example usage (uncomment to run)
# ------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("eval.csv", index_col=0)
    df = evaluate_wolfram_df(df)
    df.to_csv("eval_done.csv")