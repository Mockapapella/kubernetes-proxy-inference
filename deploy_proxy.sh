docker build -t registry.digitalocean.com/corelogic-cr-demo/proxy:latest -f docker/Dockerfile.proxy .
docker push registry.digitalocean.com/corelogic-cr-demo/proxy:latest
kubectl apply -f kubernetes/proxy.deployment.yaml
kubectl rollout restart deployment proxy-deployment
kubectl rollout status deployment proxy-deployment
