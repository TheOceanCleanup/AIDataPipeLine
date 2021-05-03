# AIDAtaPipeLine - A series of examples and utilities for Azure Machine Learning Services
# Copyright (C) 2020-2021 The Ocean Cleanup™
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import re

def set_params(path, base_config_file, new_config_file, params):
    print('Writing custom configuration file')

    with open(os.path.join(path, base_config_file)) as f:
        s = f.read()

        for param in params:
            if re.search(param, s) is not None:
                s = re.sub(param + '\s*:\s*.*',
                           param + ': ' + str(params[param][0]), s, count = params[param][1])
            else:
                raise ValueError(f"Parameter {param} not found")

    with open(os.path.join(path, new_config_file), 'w') as f:
        f.write(s)

    print('Done writing custom configuration file')
