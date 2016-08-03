import logging as log
import os
import re
import sys
from copy import copy
from tempfile import gettempdir

from setuptools import setup

DESCRIPTION = ("Sensible defaults for MSMs")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "msmpipeline/pipeline.py")
REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")

with open(str(MODULE_PATH), encoding="utf-8-sig") as source_code_file:
    SOURCE = source_code_file.read()

def make_logger(name=str(os.getpid())):
    """Build and return a Logging Logger."""
    if not sys.platform.startswith("win") and sys.stderr.isatty():
        def add_color_emit_ansi(fn):
            """Add methods we need to the class."""
            def new(*args):
                """Method overload."""
                if len(args) == 2:
                    new_args = (args[0], copy(args[1]))
                else:
                    new_args = (args[0], copy(args[1]), args[2:])
                if hasattr(args[0], 'baseFilename'):
                    return fn(*args)
                levelno = new_args[1].levelno
                if levelno >= 50:
                    color = '\x1b[31;5;7m\n '  # blinking red with black
                elif levelno >= 40:
                    color = '\x1b[31m'  # red
                elif levelno >= 30:
                    color = '\x1b[33m'  # yellow
                elif levelno >= 20:
                    color = '\x1b[32m'  # green
                elif levelno >= 10:
                    color = '\x1b[35m'  # pink
                else:
                    color = '\x1b[0m'  # normal
                try:
                    new_args[1].msg = color + str(new_args[1].msg) + ' \x1b[0m'
                except Exception as reason:
                    print(reason)  # Do not use log here.
                return fn(*new_args)
            return new
        # all non-Windows platforms support ANSI Colors so we use them
        log.StreamHandler.emit = add_color_emit_ansi(log.StreamHandler.emit)
    else:
        log.debug("Colored Logs not supported on {0}.".format(sys.platform))
    log_file = os.path.join(gettempdir(), str(name).lower().strip() + ".log")
    log.basicConfig(level=-1, filemode="w", filename=log_file,
                    format="%(levelname)s:%(asctime)s %(message)s %(lineno)s")
    log.getLogger().addHandler(log.StreamHandler(sys.stderr))
    adrs = "/dev/log" if sys.platform.startswith("lin") else "/var/run/syslog"
    try:
        handler = log.handlers.SysLogHandler(address=adrs)
    except Exception:
        log.warning("Unix SysLog Server not found,ignored Logging to SysLog.")
    else:
        log.addHandler(handler)
    log.debug("Logger created with Log file at: {0}.".format(log_file))
    return log


# Should be all UTF-8 for best results
def make_root_check_and_encoding_debug():
    """Debug and Log Encodings and Check for root/administrator,return Boolean.
    >>> make_root_check_and_encoding_debug()
    True
    """
    log.info(__doc__)
    log.debug("STDIN Encoding: {0}.".format(sys.stdin.encoding))
    log.debug("STDERR Encoding: {0}.".format(sys.stderr.encoding))
    log.debug("STDOUT Encoding:{}".format(getattr(sys.stdout, "encoding", "")))
    log.debug("Default Encoding: {0}.".format(sys.getdefaultencoding()))
    log.debug("FileSystem Encoding: {0}.".format(sys.getfilesystemencoding()))
    log.debug("PYTHONIOENCODING Encoding: {0}.".format(
        os.environ.get("PYTHONIOENCODING", None)))
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if not sys.platform.startswith("win"):  # root check
        if not os.geteuid():
            log.critical("Runing as root is not Recommended,NOT Run as root!.")
            return False
    return True


def set_process_name_and_cpu_priority():
    """Set process name and cpu priority.
    >>> set_process_name_and_cpu_priority("test_test")
    True
    """
    try:
        os.nice(19)  # smooth cpu priority
        return True
    except Exception:
        return False  # this may fail on windows and its normal, so be silent.


def find_this(search, source=SOURCE):
    """Take a string and a filename path string and return the found value."""
    log.debug("Searching for {what}.".format(what=search))
    if not search or not source:
        log.warning("Not found on source: {what}.".format(what=search))
        return ""
    return str(re.compile(r".*__{what}__ = '(.*?)'".format(
        what=search), re.S).match(source).group(1)).strip().replace("'", "")


def parse_requirements(path=REQUIREMENTS_FILE):
    """Rudimentary parser for the requirements.txt file.
    We just want to separate regular packages from links to pass them to the
    'install_requires' and 'dependency_links' params of the 'setup()'.
    """
    log.debug("Parsing Requirements from file {what}.".format(what=path))
    pkgs, links = ["pip"], []
    if not os.path.isfile(path):
        return pkgs, links
    try:
        requirements = map(str.strip, path.splitlines())
    except Exception as reason:
        log.warning(reason)
        return pkgs, links
    for req in requirements:
        if not req:
            continue
        if 'http://' in req.lower() or 'https://' in req.lower():
            links.append(req)
            name, version = re.findall("\#egg=([^\-]+)-(.+$)", req)[0]
            pkgs.append('{package}=={ver}'.format(package=name, ver=version))
        else:
            pkgs.append(req)
    log.debug("Requirements found: {what}.".format(what=(pkgs, links)))
    return pkgs, links


make_logger()
make_root_check_and_encoding_debug()
set_process_name_and_cpu_priority()
install_requires_list, dependency_links_list = parse_requirements()
log.info("Starting build of setuptools.setup().")


setup(

    name="msmpipeline",
    version=versioneer.get_version(),

    description=DESCRIPTION,
    long_description=DESCRIPTION,

    url="https://github.com/choderalab/msm-pipeline",
    license="LGPLv3+",

    author="Sonya Hanson, Steven Albanese, Josh Fass",
    author_email="{sonya.hanson, steven.albanese, josh.fass}@choderalab.org",

    include_package_data=True,
    zip_safe=True,

    extras_require={"pip": ["pip"]},
    tests_require=['pip'],
    requires=['pip'],

    install_requires=install_requires_list,
    dependency_links=dependency_links_list,

    scripts=["msmpipeline/pipeline.py"],

    keywords=['Some', 'KeyWords', 'Here'],

    classifiers=[

        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Natural Language :: English',

        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',

        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',

    ],
)


log.info("Finished build of setuptools.setup().")
