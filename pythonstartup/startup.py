"""
Python startup file — loaded via PYTHONSTARTUP environment variable.

Setup::
    export PYTHONSTARTUP=$HOME/dotfiles/pythonstartup/startup.py
"""

import cProfile
import glob
import importlib
import inspect
import io
import os
import pickle
import pstats
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from functools import wraps
from itertools import islice
from pprint import pprint

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def profile(func):
    """Decorator that profiles a function with cProfile."""

    @wraps(func)
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def timer(func):
    """Decorator that prints wall-clock execution time."""

    @wraps(func)
    def wrap(*args, **kwargs):
        ts = datetime.now()
        result = func(*args, **kwargs)
        te = datetime.now()
        print(f"Function:{func.__name__} took: {te - ts}")
        return result

    return wrap


# ---------------------------------------------------------------------------
# Shell-like utilities
# ---------------------------------------------------------------------------

EDITOR = os.environ.get("EDITOR", "vim")


def edit(target):
    """Open a file in the default editor.
    Usage:  >>> edit('myfile.py')
    """
    subprocess.Popen(f"{EDITOR} {target}", shell=True)


def openf(path):
    """Open a file or directory with the system default handler."""
    if sys.platform == "linux":
        subprocess.Popen(["xdg-open", path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        os.startfile(path)


def reimport(mod):
    """Reload a module by name or reference. Prefer importlib.reload() directly."""
    if isinstance(mod, str):
        mod = sys.modules[mod]
    return importlib.reload(mod)


def _glob(filenames):
    """Expand filename(s) with possible shell metacharacters."""
    if isinstance(filenames, str):
        return glob.glob(filenames) or [filenames]
    result = []
    for filename in filenames:
        globbed = glob.glob(filename)
        result.extend(globbed if globbed else [filename])
    return result


def ls(path="."):
    """List directory contents.
    Usage:  >>> ls()  or  >>> ls('/some/path')
    """
    return os.listdir(path)


mkdir = os.mkdir


def rm(*args):
    """Delete file(s).
    Usage:  >>> rm('file.c', 'file.h')
    """
    for item in _glob(args):
        try:
            os.remove(item)
        except OSError as err:
            print(f"{err}: {item}")


delete = rm


def rmdir(directory):
    """Remove a directory. Prompts if non-empty."""
    try:
        os.rmdir(directory)
    except OSError:
        answer = input(f"{directory} isn't empty. Delete anyway? [n] ")
        if answer and answer[0] in "Yy":
            shutil.rmtree(directory)
            print(f"{directory} deleted.")
        else:
            print(f"{directory} unharmed.")


def mv(*args):
    """Move/rename files.
    Usage:  >>> mv('src', 'dst')  or  >>> mv('a', 'b', 'dir/')
    """
    filenames = _glob(args)
    if len(filenames) < 2:
        print("Need at least two arguments")
        return
    if len(filenames) == 2:
        os.rename(filenames[0], filenames[1])
        return
    dest_dir = filenames[-1]
    if not os.path.isdir(dest_dir):
        print("Last argument needs to be a directory")
        return
    for filename in filenames[:-1]:
        os.rename(filename, os.path.join(dest_dir, filename))


def cp(*args):
    """Copy file(s).
    Usage:  >>> cp('src', 'dst')  or  >>> cp('a', 'b', 'dir/')
    """
    filenames = _glob(args)
    if len(filenames) < 2:
        print("Need at least two arguments")
        return
    if len(filenames) == 2:
        shutil.copy(filenames[0], filenames[1])
        return
    dest_dir = filenames[-1]
    if not os.path.isdir(dest_dir):
        print("Last argument needs to be a directory")
        return
    for filename in filenames[:-1]:
        shutil.copy(filename, os.path.join(dest_dir, filename))


def cpr(src, dst):
    """Recursively copy a directory tree. Symlinks are copied as links."""
    shutil.copytree(src, dst, symlinks=True)


def rsync(src, dst, *, verbose=True, archive=True, delete=False, dry_run=False):
    """Synchronize files using rsync.
    Usage:  >>> rsync('/path/to/source/', '/path/to/destination/')
    """
    cmd = ["rsync", "-r"]
    if archive:
        cmd.append("-a")
    if verbose:
        cmd.append("-v")
    if delete:
        cmd.append("--delete")
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([src, dst])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode
    except subprocess.CalledProcessError as err:
        print(f"rsync failed with error code {err.returncode}")
        print(err.stderr)
        return err.returncode
    except FileNotFoundError:
        print("rsync not found. Please ensure rsync is installed.")
        return -1


def ln(src, dst):
    """Create a symbolic link."""
    os.symlink(src, dst)


def lnh(src, dst):
    """Create a hard link."""
    os.link(src, dst)


def pwd():
    """Print and return current working directory."""
    cwd = os.getcwd()
    print(cwd)
    return cwd


cdlist: list[str] = [os.path.expanduser("~")]


def cd(directory=-1):
    """Change directory.
    Usage:
        cd('/abs/path')     change to absolute path
        cd('rel/path')      change to relative path
        cd()                list visited directories
        cd(int)             jump to directory by index from cd()
    """
    global cdlist
    if isinstance(directory, int):
        if 0 <= directory < len(cdlist):
            cd(cdlist[directory])
        else:
            pprint(list(enumerate(cdlist)))
        return
    directory = _glob(directory)[0]
    if not os.path.isdir(directory):
        print(f"{directory} is not a directory")
        return
    directory = os.path.abspath(os.path.expandvars(directory))
    if directory not in cdlist:
        cdlist.append(directory)
    os.chdir(directory)


def env():
    """List environment variables."""
    pprint(dict(os.environ))


interactive_dir_stack: list[str] = []


def pushd(directory):
    """Push current directory onto stack and cd to a new one."""
    interactive_dir_stack.append(os.getcwd())
    cd(directory)


def popd():
    """Pop directory from stack and cd to it."""
    if not interactive_dir_stack:
        print("Stack is empty")
        return
    target = interactive_dir_stack.pop()
    cd(target)
    print(target)


def syspath():
    """Print sys.path."""
    pprint(sys.path)


def which(obj):
    """Print and return the source file + line number of a module, class, function, or method.
    Usage:  >>> which(some_object)
    """
    try:
        source_file = inspect.getfile(obj)
    except (TypeError, OSError):
        print("Cannot determine source file (built-in or C extension).")
        return None

    try:
        _, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        lineno = 1

    print(f"{type(obj).__name__} from {source_file}:{lineno}")
    return (source_file, lineno)


whence = which


# ---------------------------------------------------------------------------
# Date parsing utilities
# ---------------------------------------------------------------------------

def parse_simple_date(date_str):
    """Parse 'YYYY-MM-DD' string to datetime."""
    return datetime.strptime(str(date_str), "%Y-%m-%d")


def parse_utc(date_str):
    """Parse 'YYYY-MM-DDTHH:MM:SS.ffffff+00' to datetime (drops timezone suffix)."""
    return datetime.strptime(str(date_str)[:-4], "%Y-%m-%dT%H:%M:%S.%f")


def fast_parse(df, col, parser=parse_utc, name=None):
    """Efficiently parse a date column by only parsing unique values once."""
    uniques = pd.DataFrame(df[col].unique(), columns=["_key"])
    uniques["_parsed"] = uniques["_key"].apply(parser)
    mapping = dict(zip(uniques["_key"], uniques["_parsed"]))
    target_col = name or col
    df[target_col] = df[col].map(mapping)
    return df


# ---------------------------------------------------------------------------
# File I/O shortcuts
# ---------------------------------------------------------------------------

def rcsv(path, **kwargs):
    """Shortcut for pd.read_csv()."""
    return pd.read_csv(path, **kwargs)


def rexcel(path, **kwargs):
    """Shortcut for pd.read_excel()."""
    return pd.read_excel(path, **kwargs)


def rpq(path, **kwargs):
    """Shortcut for pd.read_parquet()."""
    return pd.read_parquet(path, **kwargs)


def read(fname, nlines=10):
    """Print the first n lines of a file."""
    with open(fname, encoding="utf-8", errors="replace") as f:
        for line in islice(f, nlines):
            print(line, end="")


def savepickle(obj, filename):
    """Pickle an object to a file."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def loadpickle(filename):
    """Load a pickled object from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# String / regex utilities
# ---------------------------------------------------------------------------

def findint(s):
    """Extract the first integer from a string."""
    match = re.search(r"\d+", s)
    if match is None:
        raise ValueError(f"No integer found in {s!r}")
    return int(match.group())


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------

def qprint(df, first_n=5):
    """Quick-print the first n unique values per column."""
    for col in df.columns:
        print(f"{col}: {df[col].unique()[:first_n]}")


def findcol(df, word, ignore_case=True):
    """Find columns in a DataFrame whose names contain `word`.
    Usage:  >>> findcol(df, 'price')
    """
    cols = df.columns.tolist()
    if ignore_case:
        return [c for c in cols if word.lower() in c.lower()]
    return [c for c in cols if word in c]


findcols = find_col = find_cols = findcol


def marketshare(df, col):
    """Add a market-share (%) column based on `col`."""
    df[f"{col}_mrkshr"] = df[col] / df[col].sum() * 100
    return df


def tw_metrics(df, col, timeindex):
    """Compute a time-weighted metric for `col` grouped by `timeindex`."""
    temp = df.groupby(timeindex)[col].sum().reset_index().sort_values(by=timeindex)
    temp["time_d"] = temp[timeindex].diff().dt.total_seconds()
    temp[f"{col}_tw"] = temp[col] * temp["time_d"]
    return temp[f"{col}_tw"].sum() / temp.dropna(subset=[col])["time_d"].sum()


def winsorize_series(s, limits=(0.01, 0.01)):
    """Winsorize a Series by clipping at the given quantile limits."""
    return s.clip(
        lower=s.quantile(limits[0]),
        upper=s.quantile(1 - limits[1]),
    )


def mwinsorize(df, cols, limits=(0.01, 0.01), return_raw=False):
    """Winsorize multiple columns in-place. Optionally return a pre-winsorize copy."""
    raw = df.copy() if return_raw else None
    for col in cols:
        df[col] = winsorize_series(df[col], limits)
    return (df, raw) if return_raw else df


# ---------------------------------------------------------------------------
# Pandas display settings
# ---------------------------------------------------------------------------

PDSET_OPTIONS = {
    "mc": "display.max_columns",
    "mr": "display.max_rows",
    "cw": "display.max_colwidth",
    "w": "display.width",
}

PDSET_PRESETS = {
    "default": {"mc": 15, "mr": 50, "cw": 100, "w": 300, "d": 4},
    "l": {"mc": 150, "mr": 500, "cw": 500, "w": 1000, "d": 4},
}


def pdset(opt="default", value=-1):
    """Quick pandas display option setter.
    Presets:  pdset()  or  pdset('l')
    Single:   pdset('mc', 50)  /  pdset('mr', 200)  /  pdset('d', 2)
    Keys: mc=max_columns, mr=max_rows, cw=max_colwidth, w=width, d=decimal digits
    """
    opt = opt.lower()
    if value == -1 and opt in PDSET_PRESETS:
        preset = PDSET_PRESETS[opt]
        for key, val in preset.items():
            if key == "d":
                pd.set_option("display.float_format", f"{{:.{val}f}}".format)
            else:
                pd.set_option(PDSET_OPTIONS[key], val)
        return

    if opt == "d":
        pd.set_option("display.float_format", f"{{:.{value}f}}".format)
    elif opt in PDSET_OPTIONS:
        pd.set_option(PDSET_OPTIONS[opt], value)
    else:
        raise KeyError(f"Unknown option {opt!r}. Valid: {', '.join(PDSET_OPTIONS)} / d / presets: {', '.join(PDSET_PRESETS)}")


pdset()


# ---------------------------------------------------------------------------
# Code inspection
# ---------------------------------------------------------------------------

def source(func, limit=1000):
    """Print the source code of a function or class (up to `limit` chars)."""
    src = inspect.getsource(func)
    print(src[:limit] + ("..." if len(src) > limit else ""))


# Backward compat alias
_inspect_ = source


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(df, y, x, y2=None, plot_type="line", title=None, ascending=True, axis=True, show=True, **kwargs):
    """Plot y over x using Plotly with optional secondary y-axis.

    Parameters
    ----------
    df : pd.DataFrame
    y : str — primary y-axis column
    x : str — x-axis column
    y2 : str | None — secondary y-axis column
    plot_type : 'line' | 'scatter' | 'bar'
    title : str | None
    ascending : bool — sort by x
    axis : bool — show zero lines
    show : bool — if False, return the figure instead of displaying
    """
    import plotly.graph_objects as go

    if plot_type in ("line", "bar"):
        df = df.sort_values(by=x, ascending=ascending)

    trace_map = {
        "line": lambda y_col, name, yaxis: go.Scatter(x=df[x], y=df[y_col], name=name, mode="lines", yaxis=yaxis),
        "scatter": lambda y_col, name, yaxis: go.Scatter(x=df[x], y=df[y_col], name=name, mode="markers", yaxis=yaxis),
        "bar": lambda y_col, name, yaxis: go.Bar(x=df[x], y=df[y_col], name=name, yaxis=yaxis),
    }
    if plot_type not in trace_map:
        raise ValueError(f"Unsupported plot_type {plot_type!r}. Choose from {list(trace_map)}")

    make_trace = trace_map[plot_type]
    fig = go.Figure()
    fig.add_trace(make_trace(y, y, "y"))

    if y2 is not None:
        fig.add_trace(make_trace(y2, y2, "y2"))
        fig.update_layout(yaxis2=dict(title=y2, overlaying="y", side="right", showgrid=False))

    fig.update_layout(
        title=title or f"{y} vs {x}" + (f" with {y2}" if y2 else ""),
        xaxis_title=x,
        yaxis_title=y,
        legend=dict(x=0.01, y=0.99, borderwidth=1),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=80, r=80, t=60, b=60),
    )
    if axis:
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

    if show:
        fig.show()
    else:
        return fig


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
print("=" * 70)
print("User: Jiayuan Chen (jiayuanchen@outlook.com)")
print("=" * 70)
