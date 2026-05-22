# Luminark Infrastructure

> Local Docker Compose + Cloud Kubernetes

## Quick Start (Local)

```bash
# One command
./infra/scripts/run_local.sh up
```

Services:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LOCAL (CPU)                         │
├─────────────────────────────────────────────────────────┤
│  docker-compose.yml                                      │
│  ├── backend (CPU, PyTorch)                             │
│  ├── frontend (React/Vite)                              │
│  ├── db (PostgreSQL)                                    │
│  └── cache (Redis)                                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     CLOUD (GPU)                          │
├─────────────────────────────────────────────────────────┤
│  Kubernetes                                              │
│  ├── backend-cpu (edge workloads, autoscaling)          │
│  ├── backend-gpu (heavy workloads, NVIDIA)              │
│  ├── frontend (static + API proxy)                      │
│  └── ingress (TLS, routing)                             │
└─────────────────────────────────────────────────────────┘
```

## Local Commands

| Command | Description |
|---------|-------------|
| `./infra/scripts/run_local.sh up` | Start stack |
| `./infra/scripts/run_local.sh down` | Stop stack |
| `./infra/scripts/run_local.sh logs` | View logs |
| `./infra/scripts/run_local.sh status` | Service status |
| `./infra/scripts/run_local.sh clean` | Remove volumes |

## Kubernetes Deployment

```bash
# Deploy to cluster
kubectl apply -f infra/k8s/

# Check status
kubectl get pods -n luminark
```

### Manifest Order

1. `00-namespace.yaml` - Namespace
2. `01-config.yaml` - ConfigMap + Secrets
3. `02-backend-cpu.yaml` - CPU backend + HPA
4. `03-backend-gpu.yaml` - GPU backend + HPA
5. `04-frontend.yaml` - Frontend
6. `05-services.yaml` - Services
7. `06-ingress.yaml` - Ingress + PVC

## CPU vs GPU

| Deployment | Use Case | Resources |
|------------|----------|-----------|
| `backend-cpu` | Edge, light, offline | 2-10 replicas, CPU only |
| `backend-gpu` | Cloud, heavy, fast | 1-4 replicas, NVIDIA GPU |

## File Structure

```
infra/
├── docker/
│   ├── Dockerfile.backend      # CPU
│   ├── Dockerfile.backend-gpu  # GPU
│   └── Dockerfile.frontend
├── k8s/
│   ├── 00-namespace.yaml
│   ├── 01-config.yaml
│   ├── 02-backend-cpu.yaml
│   ├── 03-backend-gpu.yaml
│   ├── 04-frontend.yaml
│   ├── 05-services.yaml
│   └── 06-ingress.yaml
└── scripts/
    └── run_local.sh
```
