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
