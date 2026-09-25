"""mkdocs hook: silence deprecation warnings raised inside build dependencies.

Three warnings come from mkdocs-gallery 0.10.4, which is the latest release and
still calls `mkdocs.utils.warning_filter` (removed in mkdocs 1.2) and the
`ast.Str` / `ast.Str.s` APIs (removed in Python 3.14). None is actionable from
this repository, and all three are raised on every build.

The `ast` warnings mark a real incompatibility: mkdocs-gallery 0.10.4 will fail
on Python 3.14. Filtering them here keeps the build output readable; it does not
remove the incompatibility. The project tests on 3.10 through 3.13.

`JUPYTER_PLATFORM_DIRS` is set because jupyter_core warns when it is unset and
will change its default in v6.
"""

import os
import warnings

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

for _message, _module in (
    (r"warning_filter doesn't do anything", r"mkdocs_gallery\..*"),
    (r"ast\.Str is deprecated", r"mkdocs_gallery\..*"),
    (r"Attribute s is deprecated", r"mkdocs_gallery\..*"),
    (r"Jupyter is migrating its paths", r"jupyter_core\..*"),
):
    warnings.filterwarnings("ignore", message=_message, category=DeprecationWarning, module=_module)
