#!/bin/bash
### Remove the monitor tools
kubectl delete -f Inside_SUT_deployment.yaml
kubectl delete ns monitor

### Remove the storage and dashboard resource
kubectl delete -f AIOps_deploy.yaml
kubectl delete ns aiops

### Remove the Machine learning resource
kubectl delete -f torch.yaml
kubectl delete ns aiops