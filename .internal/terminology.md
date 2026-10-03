## Terminology

### General

| Term | Definition |
|------|------------|
| Model | The LLM to be deployed. |
| Hardware | The computing device or devices on which the model is deployed. |
| Model–hardware pair | A specific supported combination of an LLM model and the hardware on which it is deployed. |
| Deployment configuration | The vLLM options used to start the deployment server for a particular model–hardware pair. |
| Supported pair | A model–hardware pair with a corresponding deployment configuration in the deployment-configuration library. |
| Unsupported pair | A model–hardware pair without a corresponding deployment configuration in the deployment-configuration library. |
| Deployment configuration library | The finite collection of deployment configurations for the supported model–hardware pairs. |
| Validation service | The independently managed service that invokes a deployment entry point and evaluates the resulting deployment against the acceptance criteria. |
| Validation profile | The validation service configuration that specifies the acceptance criteria and validation settings used for a validation request. |
| Full validation | The time-consuming validation run performed through the `/validate` endpoint. |
| Partial validation | The quick validation run performed through the `/validate_partial` endpoint for debugging and diagnostic purposes. |
| Solution override | An optional argument supplied to the deployment entry point through the validation request. |
| Acceptance criterion | A condition evaluated by the validation service using one or more metrics. |
| Metric | A value computed by the validation service to evaluate an acceptance criterion. |
| Hydra | The configuration framework used to invoke the deployment entry point and load deployment configurations. |
| vLLM | The deployment backend used to run the deployment server. |
| Baseline deployment strategy | A deployment using the original model precision and no extra deployment options. |
| Accuracy loss | The difference in accuracy between a deployment configuration and the baseline deployment strategy, expressed as a percentage. |

### Deployment CLI

| Term | Definition |
|------|------------|
| Deployment CLI | The command-line entry point that receives a model and hardware, selects a deployment configuration, and starts vLLM. |
| Model identifier | The identifier used to select the model to deploy. |
| Hardware identifier | The identifier used to select the hardware-specific deployment configuration. |
| Deployment entry point | The repository script invoked by the validation service to start a deployment. |
| Unsupported-pair error | The error reported when the requested model–hardware pair has no deployment configuration. |

### Deployment server

| Term | Definition |
|------|------------|
| Deployment server | A running vLLM service configured for a selected model–hardware pair that accepts inference requests. |
| OpenAI-compatible API | The API exposed by the deployment server for serving inference requests in the vLLM-compatible format. |
| Readiness check | A check used to determine whether the deployment server is ready to accept requests. |
| `/health` endpoint | The deployment server endpoint polled by the validation service; HTTP 200 indicates readiness. |
| `/v1/models` endpoint | The deployment server endpoint used to check that the expected model is loaded. |
