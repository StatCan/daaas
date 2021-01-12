## v1.0.2 @ [statcan/minio-operator](https://github.com/statcan/minio-operator/releases/tag/v1.0.2)

Features / Updates:

- Update for MinIO Server [RELEASE.2021-01-05T05-22-38Z](https://github.com/minio/minio/tree/RELEASE.2021-01-05T05-22-38Z)
  - Updated to RELEASE.2021-01-05T05-22-38Z of the MinIO Server image
  - Important list caching improvements
  - Major reduction in crawler memory usage
  - Important performance improvements with list caching and memory consumption
  - Do not use large buffers if not necessary
  - Update MinIO operator to v3.0.29
- Patched MinIO Operator with binary lookup fix
- Pulling MinIO from k8scc01covidacr.azurecr.io with ImagePullSecret now

> **Note**: N/A

## v1.0.0 @ [statcan/goofys-injector](https://github.com/statcan/goofys-injector/releases/tag/v1.0.0)

Features / Updates:

- Add container Index logic to only add volume mounts to specific container
- Add MinIO mounts to Argo Workflows
- Add the new tenants (standard, and premium)
- Inject MinIO volume mounts using goofys

> **Note**: N/A

## v1.0.1 @ [statcan/kubeflow-containers](https://github.com/statcan/kubeflow-containers/releases/tag/v1.0.1)

Features / Updates:

- Added back Kale extension [v0.6.0](https://github.com/kubeflow-kale/kale/releases/tag/v0.6.0)
  - Added Kale namespace logic via new NB_NAMESPACE parameter
  - Configure KFP via `$HOME/.config/kfp/context.json` file
- Use short hashes for containers
- LANG based on KF_LANG environment variable

> **Note**: N/A

## v1.0.1 @ [statcan/kubeflow-manifest](https://github.com/statcan/kubeflow-manifest/releases/tag/v1.0.1)

Features / Updates:

- Added back Kale extension [v0.6.0](https://github.com/kubeflow-kale/kale/releases/tag/v0.6.0)
- Use short hashes for containers

> **Note**: N/A

## v1.0.1 @ [statcan/mlflow-operator](https://github.com/statcan/mlflow-operator/releases/tag/v1.0.1)

Features / Updates:

- Update MLFlow to [v1.13.0](https://github.com/mlflow/mlflow/releases/tag/v1.13.0)
  - Updated to latest stable of MLFlow v1.13.0
- Re-architected folder structure and aligned with minio-operator for naming scheme

> **Note**: N/A

## v1.0.0 @ [statcan/kubeflow-mlops](https://github.com/statcan/kubeflow-mlops/releases/tag/v1.0.0)

Features / Updates:

- Updated CI / CD to support experiments now running in user level namespaces
- Updated the MLFlow python packages to 1.13.0
- Updated Tensorflow container image and python packages to 2.2.1

> **Note**: Right now the pipeline is executing as a specialized user, looking into how to paramterize to Users namespace on CI run

## v1.0.0 @ [statcan/kubeflow-containers](https://github.com/statcan/kubeflow-containers/releases/tag/v1.0.0)

Features / Updates:

- Update to jupyter/datascience-notebook
  - Update base image to jupyter/datascience-notebook:42f4c82a07ff
  - Refined container image list to JupyterLab-CPU, JupyterLab-PyTorch, JupyterLab-Tensorflow, and RStudio
  - Updated to Docker Bits hierarchical structure
- Update Visual Studio code-server to [v3.8.0](https://github.com/cdr/code-server/releases/tag/v3.8.0)
  - Added default extensions Python and R
  - Revert to Dark Theme

> **Note**: N/A

## v1.0.0 @ [statcan/kubeflow-manifest](https://github.com/statcan/kubeflow-manifest/releases/tag/v1.0.0)

Features / Updates:

- Update to Kubeflow [v1.2.0](https://github.com/kubeflow/kubeflow/releases/tag/v1.2.0)
  - Updated to v1.2.0 of Kubeflow
  - Updated to new Kustomize format [kubeflow-5440](https://github.com/kubeflow/kubeflow/issues/5440)
  - Enabled Multi User pipelines
  - Disabled endpoints for GKE

> **Note**: As of this release we now have Multi-User Pipelines support. Please see the following issue for more information at [kubeflow-5440](https://github.com/kubeflow/kubeflow/issues/5440)

## v1.0.0 @ [statcan/mlflow-operator](https://github.com/statcan/mlflow-operator/releases/tag/v1.0.0)

Features / Updates:

- Update MLFlow to [v1.12.1](https://github.com/mlflow/mlflow/releases/tag/v1.12.1)
  - Updated to latest stable of MLFlow v1.12.1
  - Updated all of the azure dependencies (data lake, blob account, file share, and queue)
  - Updated the PostgreSQL dependency

> **Note**: Added logic to upgrade the indexed db between version bumps. Please see the following commit for more information at [b901376](https://github.com/StatCan/mlflow-operator/commit/b90137648d7d8c0751eed17b0a4ce5e637400f8a)

## v1.0.0 @ [statcan/minio-operator](https://github.com/statcan/minio-operator/releases/tag/v1.0.0)

Features / Updates:

- Update for MinIO Operator [v3.0.29](https://github.com/minio/operator/releases/tag/v3.0.29)
  - Updated to v3.0.29 of the MinIO Operator image
  - Deprecation of older MinIO tenants in favor of standard and premium
  - Addition of new Open Policy Agent rules
  - Synced the old tenants to the new tenants
- Update for MinIO Console [v0.4.6](https://github.com/minio/console/releases/tag/v0.4.6)
  - Updated to v0.4.6 of the MinIO Console image
- Update for MinIO Server [RELEASE.2020-12-10T01-54-29Z](https://github.com/minio/minio/tree/RELEASE.2020-12-10T01-54-29Z)
  - Updated to RELEASE.2020-12-10T01-54-29Z of the MinIO Server image

> **Note**: This was unfortunately a breaking change as we were initially using a very early alpha release. Please see the following issue for more information at [daaas-300](https://github.com/StatCan/daaas/issues/300)
