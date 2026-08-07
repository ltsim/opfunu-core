#!/usr/bin/env python
# Created by "Thieu" at 06:25, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from .cec2005 import *  # noqa: F403
from .cec2008 import *  # noqa: F403
from .cec2010 import *  # noqa: F403
from .cec2013 import *  # noqa: F403
from .cec2014 import *  # noqa: F403
from .cec2015 import *  # noqa: F403
from .cec2017 import *  # noqa: F403
from .cec2019 import *  # noqa: F403
from .cec2020 import *  # noqa: F403
from .cec2021 import *  # noqa: F403
from .cec2022 import *  # noqa: F403

__all__ = [s for s in dir() if not s.startswith('_')]
