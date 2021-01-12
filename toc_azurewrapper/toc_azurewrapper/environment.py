from azureml.core import Environment
from azureml.exceptions import AzureMLException


def get_environment(workspace, name, pip_requirements=None,
                    conda_specification=None, conda_env=None,
                    docker_image=None, docker_file=None,
                    override=False, add_inference_stack=False):
    """
    Get an Azure ML environment from PIP or Conda. From:
    - pip_requirements
    - conda_specification
    - conda_env
    at most one can be provided. If none is provided, it is assumed that the
    requirements are taken care of by the user.

    From:
    - docker_image
    - docker_file
    at most one can be provided. If none is provided, the base Azure image is
    used.

    :params workspace:              The Azure ML workspace to look for existing
                                    environments.
    :params name:                   Name for this environment
    :params pip_requirements:       Path to the pip requirements file
    :params conda_specifidation:    Path to the conda specification file
    :params conda_env:              Name of the conda environment to use
    :params docker_image:           Base the image off an existing docker image
    :params docker_file:            Base the image off a Dockerfile.
    :params override:               Create a new environment with this name,
                                    regardless of if one already exists.
    :params add_inference_stack:    Add a stack that enables this environment
                                    for inference. "latest" is a valid option.
                                    Set to False to not add this.
    :returns:                       Azure ML environment or None in case of
                                    failure
    """
    if not override:
        try:
            env = Environment.get(workspace, name)

            print("Existing environment found, using that")
            return env
        except:
            print("No environment with that name found, creating new one")

    # Validate at most one of pip_requirements, conda_specification, conda_env
    # is provided
    if sum([1 for x in [pip_requirements, conda_specification, conda_env]
            if x is not None]) > 1:
        print("Provide at most 1 of pip_requirements, conda_specification, "
              "conda_env")
        return None

    # Validate that at most one of docker_image, docker_file is
    # provided
    if sum([1 for x in [docker_image, docker_file]
            if x is not None]) > 1:
        print("Provide at most 1 of docker_image, docker_file")
        return None

    if pip_requirements is not None:
        env = Environment.from_pip_requirements(name, pip_requirements)
    elif conda_specification is not None:
        env = Environment.from_conda_specification(name, conda_specification)
    elif conda_env is not None:
        env = Environment.from_existing_conda_environment(name, conda_env)
    else:
        env = Environment(name)
        env.python.user_managed_dependencies = True

    if docker_file is not None:
        env.docker.enabled = True
        env.docker.base_image = None
        env.docker.base_dockerfile = docker_file
    elif docker_image is not None:
        env.docker.enabled = True
        env.docker.base_image = docker_image

    if add_inference_stack:
        env.inferencing_stack_version = add_inference_stack

    # Register environment
    env.register(workspace=workspace)

    return env
