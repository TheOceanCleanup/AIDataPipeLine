from azureml.core import Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice, Webservice, \
    AciWebservice, AksWebservice
from azureml.exceptions import WebserviceException


def deploy(workspace, name, model, script, source_directory, environment=None,
        target='local', cpu_cores=1, memory_gb=1, compute_target_name=None):
    inference_config = InferenceConfig(
        entry_script=script,
        source_directory=source_directory,
        environment=environment
    )

    if target == 'local':
        deployment_config = LocalWebservice.deploy_configuration(port=8890)
    elif target == 'aci':
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb
        )
    elif target == 'aks':
        if compute_target_name is None:
            print("compute_target_name required when target='aks'")
            return None
        deployment_config = AksWebservice.deploy_configuration(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            compute_target_name=compute_target_name,
            auth_enabled=False
        )

    try:
        service = Webservice(workspace, name)
    except WebserviceException:
        service = None

    if service is None:
        service = Model.deploy(
            workspace,
            name,
            [model],
            inference_config,
            deployment_config
        )
    else:
        print(
            "Existing service with that name found, updating InferenceConfig\n"
            "If you meant to redeploy or change the deployment option, first "
            "delete the existing service."
        )
        service.update(
            models=[model],
            inference_config=inference_config)
    return service
