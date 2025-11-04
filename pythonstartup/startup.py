"""
My Python startup file, carefully gathered from different sources (see below)
Get code from Github::
    git clone https://github.com/jezdez/python-startup.git ~/.python
Put this in your shell profile::
    export PYTHONSTARTUP=$HOME/.python/startup.py
In case you haven't saved these files in $HOME/.python make sure to set
PYTHONUSERDIR approppriately, too::
    export PYTHONUSERDIR=/path/to/dir
"""
# python-startup.py
# Author: Nathan Gray, based on interactive.py by Robin Friedrich and an
#           evil innate desire to customize things.
# E-Mail: n8gray@caltech.edu
#
# Version: 0.6

# These modules are always nice to have in the namespace

############################################################################
# Below this is Robin Friedrich's interactive.py with some edits to decrea7se
# namespace pollution and change the help functionality
# NG
#
# Also enhanced 'which' to return filename/lineno
# Patch from Stephan Fiedler to allow multiple args to ls variants
# NG 10/21/01  --  Corrected a bug in _glob
#
########################### interactive.py ###########################
#  """Functions for doing shellish things in Python interactive mode.
#
#     Meant to be imported at startup via environment:
#       setenv PYTHONSTARTUP $HOME/easy.py
#       or
#       export PYTHONSTARTUP=$HOME/easy.py
#
#     - Robin Friedrich
#  """


import functools
import cProfile, pstats, io
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import types
from itertools import islice
from pprint import pprint
import pickle

import numpy as np
import pandas as pd
from functools import wraps
from datetime import datetime

# environment settings:
#pd.set_option('display.max_column', None)
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_colwidth', 1000)
#pd.set_option('display.width', 1000)
#pd.set_option('display.float_format', lambda x: '%.2f' % x)

def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

try:
    from pydoc import help
except ImportError:
    def help(*objects):
        """Print doc strings for object(s).
        Usage:  >>> help(object, [obj2, objN])  (brackets mean [optional] argument)
        """
        if len(objects) == 0:
            help(help)
            return
        for obj in objects:
            try:
                print('****', obj.__name__, '****')
                print(obj.__doc__)
            except AttributeError:
                print(obj, 'has no __doc__ attribute')
                print

try:
    from collections import defaultdict
except ImportError:
    pass

# home = os.path.expandvars('$HOME')
# user_dir = os.path.join(home, os.environ.get("PYTHONUSERDIR", ".python"))
# sys.path.append(user_dir)

##### Some settings you may want to change #####
# Define the editor used by the edit() function. Try to use the editor
# defined in the Unix environment, or default to vi if not set.
# (patch due to Stephan Fiedler)
#
# %(lineno)s gets replaced by the line number.  Ditto %(fname)s the filename
EDITOR = os.environ.get('EDITOR', 'vim')
editorbase = EDITOR.split()[0]
if editorbase in ['nedit', 'nc', 'ncl', 'emacs', 'emacsclient', 'xemacs']:
    # We know these editors supoprt a linenumber argument
    EDITOR = EDITOR + ' +%(lineno)s %(fname)s &'
elif editorbase in ['vi', 'vim', 'jed']:
    # Don't want to run vi in the background!
    # If your editor requires a terminal (e.g. joe) use this as a template
    EDITOR = 'xterm -e ' + EDITOR + ' +%(lineno)s %(fname)s &'
else:
    # Guess that the editor only supports the filename
    EDITOR = EDITOR + ' %(fname)s &'
del editorbase

# The place to store your command history between sessions
# histfile = os.path.join(user_dir, "history")

# Functions automatically added to the builtins namespace so that you can
# use them in the debugger and other unusual environments
autobuiltins = ['edit', 'which', 'ls', 'cd', 'mv', 'cp', 'rm', 'help', 'rmdir',
                'ln', 'pwd', 'pushd', 'popd', 'env', 'mkdir']

# LazyPython only works for Python versions 2.1 and above
# try:
#     # Try to use LazyPython
#     from LazyPython import LazyPython
#
#     sys.excepthook = LazyPython()
# except ImportError:
#     pass
#
# try:
#      Try to set up command history completion/saving/reloading
#     import readline, atexit, rlcompleter
#
#     readline.parse_and_bind('tab: complete')
#     try:
#         readline.read_history_file(histfile)
#     except IOError:
#         pass  # It doesn't exist yet.
#
#
#     def savehist():
#         try:
#             global histfile
#             readline.write_history_file(histfile)
#         except:
#             print('Unable to save Python command history')
#
#
#     atexit.register(savehist)
#     del atexit
# except ImportError:
#     pass


# Make an "edit" command that sends you to the right file *and line number*
# to edit a module, class, method, or function!
# Note that this relies on my enhanced version of which().
# def edit(object, editor=EDITOR):
#     """Edit the source file from which a module, class, method, or function
#     was imported.
#     Usage:  >>> edit(mysteryObject)
#     """
#
#     if type(object) is type(""):
#         fname = object;
#         lineno = 1
#         print(editor % locals())
#         subprocess.Popen(editor % locals(), shell=True)
#         return
#
#     ret = which(object)
#     if not ret:
#         print("Can't edit that!")
#         return
#     fname, lineno = ret
#     if fname[-4:] == '.pyc' or fname[-4:] == '.pyo':
#         fname = fname[:-1]
#     print(editor % locals())
#     subprocess.Popen(editor % locals(), shell=True)

def edit(object):
    """Edit the source file from which a module, class, method, or function
    was imported.
    Usage:  >>> edit(mysteryObject)
    """
    subprocess.Popen(f'subl + {object}', shell=True)


def openf(directory):
    os.startfile(directory)


def reimport(mod, locals=None):
    if isinstance(mod, str):
        modname = mod
    else:
        modname = mod.__name__
    sys.modules[modname] = None
    del sys.modules[modname]
    new_mod = __import__(modname)
    if locals is not None:
        locals[modname] = new_mod
    return new_mod


def _glob(filenames):
    """Expand a filename or sequence of filenames with possible
    shell metacharacters to a list of valid filenames.
    Ex:  _glob(('*.py*',)) == ['able.py','baker.py','charlie.py']
    """
    if type(filenames) is str:
        return glob.glob(filenames)
    flist = []
    for filename in filenames:
        globbed = glob.glob(filename)
        if globbed:
            for file in globbed:
                flist.append(file)
        else:
            flist.append(filename)
    return flist


def _expandpath(d):
    """Convert a relative path to an absolute path.
    """
    return os.path.join(os.getcwd(), os.path.expandvars(d))


lsdir = os.listdir
mkdir = os.mkdir


def rm(*args):
    """Delete a file or files.
    Usage:  >>> rm('file.c' [, 'file.h'])  (brackets mean [optional] argument)
    Alias: delete
    """
    filenames = _glob(args)
    for item in filenames:
        try:
            os.remove(item)
        except OSError as detail:
            print(f'{detail} : {item}')


delete = rm


def rmdir(directory):
    """Remove a directory.
    Usage:  >>> rmdir('dirname')
    If the directory isn't empty, can recursively delete all sub-files.
    """
    try:
        os.rmdir(directory)
    except os.error:
        # directory wasn't empty
        answer = raw_input(directory + " isn't empty. Delete anyway?[n] ")
        if answer and answer[0] in 'Yy':
            subprocess.Popen('rm -rf %s' % directory, shell=True)
            print(directory + ' Deleted.')
        else:
            print(directory + ' Unharmed.')


def mv(*args):
    """Move files within a filesystem.
    Usage:  >>> mv('file1', ['fileN',] 'fileordir')
    If two arguments - both must be files
    If more arguments - last argument must be a directory
    """
    filenames = _glob(args)
    nfilenames = len(filenames)
    if nfilenames < 2:
        print('Need at least two arguments')
    elif nfilenames == 2:
        try:
            os.rename(filenames[0], filenames[1])
        except OSError as detail:
            print("%s: %s" % (detail[1], filenames[1]))
    else:
        for filename in filenames[:-1]:
            try:
                dest = filenames[-1] + '/' + filename
                if not os.path.isdir(filenames[-1]):
                    print('Last argument needs to be a directory')
                    return
                os.rename(filename, dest)
            except OSError as detail:
                print("%s: %s" % (detail[1], filename))


def cp(*args):
    """Copy files along with their mode bits.
    Usage:  >>> cp('file1', ['fileN',] 'fileordir')
    If two arguments - both must be files
    If more arguments - last argument must be a directory
    """
    filenames = _glob(args)
    nfilenames = len(filenames)
    if nfilenames < 2:
        print('Need at least two arguments')
    elif nfilenames == 2:
        try:
            shutil.copy(filenames[0], filenames[1])
        except OSError as detail:
            print("%s: %s" % (detail[1], filenames[1]))
    else:
        for filename in filenames[:-1]:
            try:
                dest = filenames[-1] + '/' + filename
                if not os.path.isdir(filenames[-1]):
                    print('Last argument needs to be a directory')
                    return
                shutil.copy(filename, dest)
            except OSError as detail:
                print("%s: %s" % (detail[1], filename))


def cpr(src, dst):
    """Recursively copy a directory tree to a new location
    Usage:  >>> cpr('directory0', 'newdirectory')
    Symbolic links are copied as links not source files.
    """
    shutil.copytree(src, dst)


def ln(src, dst):
    """Create a symbolic link.
    Usage:  >>> ln('existingfile', 'newlink')
    """
    os.symlink(src, dst)


def lnh(src, dst):
    """Create a hard file system link.
    Usage:  >>> ln('existingfile', 'newlink')
    """
    os.link(src, dst)

def pwd():
    """Print current working directory path.
    Usage:  >>> pwd()
    """
    print os.getcwd()


def cd(directory=-1):
    """Change directory. Environment variables are expanded.
    Usage:
    cd('rel/$work/dir') change to a directory relative to your own
    cd('/abs/path')     change to an absolute directory path
    cd()                list directories you've been in
    cd(int)             integer from cd() listing, jump to that directory
    """
    global cdlist
    if type(directory) is int:
        if directory in range(len(cdlist)):
            cd(cdlist[directory])
            return
        else:
            pprint(cdlist)
            return
    directory = _glob(directory)[0]
    if not os.path.isdir(directory):
        print(directory + ' is not a directory')
        return
    directory = _expandpath(directory)
    if directory not in cdlist:
        cdlist.append(directory)
    os.chdir(directory)


def env():
    """List environment variables.
    Usage:  >>> env()
    """
    # unfortunately environ is an instance not a dictionary
    pprint(os.environ)


interactive_dir_stack = []


def pushd(directory):
    """Place the current dir on stack and change directory.
    Usage:  >>> pushd(['dirname'])   (brackets mean [optional] argument)
                pushd()  goes home.
    """
    global interactive_dir_stack
    interactive_dir_stack.append(os.getcwd())
    cd(directory)


def popd():
    """Change to directory popped off the top of the stack.
    Usage:  >>> popd()
    """
    global interactive_dir_stack
    try:
        cd(interactive_dir_stack[-1])
        print(interactive_dir_stack[-1])
        del interactive_dir_stack[-1]
    except IndexError:
        print('Stack is empty')


def syspath():
    """Print the Python path.
    Usage:  >>> syspath()
    """
    import sys
    pprint(sys.path)


def which(object):
    """Print the source file from which a module, class, function, or method
    was imported.

    Usage:    >>> which(mysteryObject)
    Returns:  Tuple with (file_name, line_number) of source file, or None if
              no source file exists
    Alias:    whence
    """
    object_type = type(object)
    if object_type is types.ModuleType:
        if hasattr(object, '__file__'):
            print('Module from', object.__file__)
            return (object.__file__, 1)
        else:
            print('Built-in module.')
    elif object_type is types.ClassType:
        if object.__module__ == '__main__':
            print('Built-in class or class loaded from $PYTHONSTARTUP')
        else:
            print('Class', object.__name__, 'from', \
                  sys.modules[object.__module__].__file__)
            # Send you to the first line of the __init__ method
            return (sys.modules[object.__module__].__file__,
                    object.__init__.im_func.func_code.co_firstlineno)
    elif object_type in (types.BuiltinFunctionType, types.BuiltinMethodType):
        print("Built-in or extension function/method.")
    elif object_type is types.FunctionType:
        print('Function from', object.func_code.co_filename)
        return (object.func_code.co_filename, object.func_code.co_firstlineno)
    elif object_type is types.MethodType:
        print('Method of class', object.im_class.__name__, 'from', )
        fname = sys.modules[object.im_class.__module__].__file__
        print(fname)
        return (fname, object.im_func.func_code.co_firstlineno)
    else:
        print("argument is not a module or function.")
    return None


whence = which


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
def parse_simple_date(date, **kwargs):
    return datetime.strptime(str(date), '%Y-%m-%d')


def __ParseUTC__(d):
    return datetime.strptime(str(d)[:-4], '%Y-%m-%dT%H:%M:%S.%f')


def fast_parse(df, col, parser=__ParseUTC__, name=None):
    '''
    '''
    dt = pd.DataFrame(df[col].unique())
    dt.columns = [col + '_tmp']
    dt[col] = dt[col + '_tmp'].apply(parser)
    date_dict = dt.set_index(col + '_tmp').to_dict()
    if name == None:
        df[col] = df[col].map(date_dict[col])
    else:
        df[name] = df[col].map(date_dict[col])
    return df

def rcsv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def rexcel(path, **kwargs):
    return pd.read_excel(path, **kwargs)


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f'Function:{f.__name__} with args: took: {te - ts} seconds')
        return result

    return wrap


def folder_space(_workDir_: str, subfolder_name: str, local_folder: bool = True):
    '''
    :param name: folder name
    :return: folder path
    '''

    if _workDir_ == '':
        CurrDir = os.getcwd().replace('\\', '/') + '/'
        workDir = CurrDir if local_folder else CurrDir + 'New folder/'
    else:
        workDir = _workDir_ if _workDir_[-1] == '/' else _workDir_ + '/'
    #     filename = workDir + subfolder_name + '/'
    # else:
    #     filename = filename_temp

    filename = workDir + subfolder_name + '/'

    try:
        os.makedirs(filename)
        # print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        # print("Directory " , name ,  " already exists")
        pass
    return filename


def getd(device='HPC'):
    if device == 'HPC' or device == 0:
        return 'C:/Users/PC-user/Desktop/'
    elif device == 'UNSW' or device == 1:
        return 'C:/Users/z3446244/Desktop/'
    elif device == 'linux' or device == 2:
        return '/home/shinc/'
    elif device == 'lib' or device == 3:
        return 'D:/Anaconda/mylib/'
    elif device == 'linuxd' or device == 4:
        return '/run/media/shinc/8C36A64636A6315E/'
    elif device == 'pych' or device == 5:
        return '/run/media/shinc/8C36A64636A6315E/PycharmProjects/'
    else:
        print('Please enter HPC/UNSW/MAC')
        return os.getcwd()

device = sys.platform
workDir = getd(device)
cdlist = [getd(i) for i in range(0, 6)]


def read(fname, nlines=10):
    '''read any file for first n line'''
    try:
        with open(fname) as f:
            for line in islice(f, nlines):
                print(line)
    except:
        with open(fname, encoding="utf8") as f:
            for line in islice(f, nlines):
                print(line)


def findint(s):
    '''
    find integer from strings
    :param s:
    :return:
    '''
    return int(re.search(r'\d+', s).group())

def savepickle(var, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var, f)


def loadpickle(filename):
    with open(filename, 'rb') as f:
        temp = pickle.load(f)
    return temp


def qprint(df, first_n=5):
    print([{c: df[c].unique()[0:first_n]} for c in df.columns.tolist()])


def marketshare(df, col):
    '''
    calculate market share based on col
    '''
    total_sum = df[col].sum()
    df[col + '_mrkshr'] = df[col] / total_sum * 100
    return df


def tw_metrics(df, col, timeindex):
    temp = df.groupby(timeindex)[col].sum().reset_index()
    temp = temp.sort_values(by=timeindex)
    temp['time_d'] = (temp[timeindex].diff(1)).dt.total_seconds()
    temp[col + '_tw'] = temp[col] * temp['time_d']
    tw_col = temp[col + '_tw'].sum() / temp.dropna(subset=[col])['time_d'].sum()
    return tw_col



def findcol(df, word, ignore=True):
    cols=df.columns.tolist()
    if ignore:
        cols_temp = [x.lower() for x in cols]
        word=word.lower()
    return [cols[i] for i,x in enumerate(cols_temp) if word in x]
findcols = findcol
find_col = findcol
find_cols = findcol

def _set_digit(digit):
    pd.set_option('display.max_column', None)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)



def winsorize_with_pandas(s, limits):
    """
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array,
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'),
                  upper=s.quantile(1 - limits[1], interpolation='higher'))


def mwinsorize(df, cols, limits=[0.01, 0.01], return_raw=False):
    '''
    winsorize using pandas
    :param df:
    :param col:
    :param limits:
    :return:
    '''

    if return_raw:
        raw = df.copy()

    for col in cols:
        df[col] = winsorize_with_pandas(df[col], limits)

    if return_raw:
        return df, raw
    else:
        return df


def pdset(opt = 'default', value = -1):
    '''
    Quick way to set pandas option
    :param opt: pandas option
    'mc': pd.set_option('display.max_column', value)
    'mr': pd.set_option('display.max_rows', value)
    'cw': pd.set_option('display.max_colwidth', value)
    'w': pd.set_option('display.width', value)
    'd': pd.set_option('display.float_format', lambda x: f'%.{value}f' % x)
    :param value: optioin value
    '''
    opt = opt.lower()

    # default setting
    if opt == 'default' and value == -1:
        pd.set_option('display.max_column', 15)
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_colwidth', 100)
        pd.set_option('display.width', 300)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
    elif opt == 'l' and value == -1:
        pd.set_option('display.max_column', 150)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_colwidth', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)


    else:
        if opt == 'mc':
            pd.set_option('display.max_column', value)
            print (f'pd.set_option(display.max_column, {value})')
        elif opt == 'mr':
            pd.set_option('display.max_rows', value)
            print (f'pd.set_option(display.max_rows, {value})')
        elif opt == 'cw':
            pd.set_option('display.max_colwidth', value)
            print (f'pd.set_option(display.max_colwidth, {value})')
        elif opt == 'w':
            pd.set_option('display.width', value)
            print (f'pd.set_option(display.width, {value})')
        elif opt == 'd':
            pd.set_option('display.float_format', lambda x: f'%.{value}f' % x)
            print (f'pd.set_option(display.float_format, lambda x: %.{value}f % x)')
        else:
            raise KeyError('No matching option, please check the option keywords')
pdset()

import inspect

def _inspect_(func, limit = 1000):
    '''
    Inspect function or module
    :param func: function want to inspect
    :param limit: the number of character want to view
    :return: None
    '''
    lines = inspect.getsource(func)
    print(lines[0:limit] + '...')


print('=' * 70)
print('User: Jiayuan Chen (jiayuanchen@outlook.com)')
print('=' * 70)

def plot(df, y, x, y2=None, plot_type='line', title=None):
    """
    Plot y over x using Plotly with optional secondary y-axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Data source.
    y : str
        Column name for the primary y-axis.
    x : str
        Column name for the x-axis.
    y2 : str, optional
        Column name for the secondary y-axis (default is None).
    plot_type : str, optional
        Type of plot ('line', 'scatter', or 'bar'), default is 'line'.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The configured Plotly figure.
    """

    # Choose trace constructor based on type
    def make_trace(y_col, name, yaxis='y', color=None):
        if plot_type == 'line':
            return go.Scatter(x=df[x], y=df[y_col], name=name, mode='lines', yaxis=yaxis)
        elif plot_type == 'scatter':
            return go.Scatter(x=df[x], y=df[y_col], name=name, mode='markers', yaxis=yaxis)
        elif plot_type == 'bar':
            return go.Bar(x=df[x], y=df[y_col], name=name, yaxis=yaxis)
        else:
            raise ValueError(f"Unsupported plot_type '{plot_type}'. Choose from ['line', 'scatter', 'bar'].")

    # Create figure and add primary trace
    fig = go.Figure()
    fig.add_trace(make_trace(y, name=y, yaxis='y'))

    # Add secondary y-axis if provided
    if y2 is not None:
        fig.add_trace(make_trace(y2, name=y2, yaxis='y2'))
        fig.update_layout(
            yaxis2=dict(
                title=y2,
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

    # Layout and formatting
    fig.update_layout(
        title=title or f"{y} vs {x}" + (f" with {y2}" if y2 else ""),
        xaxis_title=x,
        yaxis_title=y,
        legend=dict(x=0.01, y=0.99, borderwidth=1),
        template="plotly_white",
        hovermode='x unified',
        margin=dict(l=80, r=80, t=60, b=60)
    )
    
    fig.show()


#
