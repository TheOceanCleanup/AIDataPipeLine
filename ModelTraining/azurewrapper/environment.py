from azureml.core import Environment
from azureml.exceptions import AzureMLException


def get_environment(workspace, name, pip_requirements=None,
                    conda_specification=None, conda_env=None, override=False):
    """
    Get an Azure ML environment from PIP or Conda.

    :params workspace:              The Azure ML workspace to look for existing
                                    environments.
    :params name:                   Name for this environment
    :params pip_requirements:       Path to the pip requirements file
    :params conda_specifidation:    Path to the conda specification file
    :params conda_env:              Name of the conda environment to use
    :params override:               Create a new environment with this name,
                                    regardless of if one already exists.
    :returns:                       Azure ML environment or None in case of
                                    failure
    """
    if not override:
        env = Environment.get(workspace, name)
        if env is None:
            print("No environment with that name found, creating new one")
        else:
            return env

    # Validate only one of pip_requirements, conda_specification, conda_env is
    # provided
    if sum([1 for x in [pip_requirements, conda_specification, conda_env]
            if x is not None]) != 1:
        print("Provide exactly 1 of pip_requirements, conda_specification, "
              "conda_env")
        return None

    if pip_requirements is not None:
        return Environment.from_pip_requirements(name, pip_requirements)
    if conda_specification is not None:
        return Environment.from_conda_specification(name, conda_specification)
    if conda_env is not None:
        return Environment.from_existing_conda_environment(name, conda_env)

    print("This is embarrassing, you shouldn't be here")
    return None
