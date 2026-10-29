# Nevrine AWS Cloud Deployment & Production Runbook

This guide details launching and running the Nevrine Neural Acceleration Engine on **AWS EC2 g4dn.xlarge** instances featuring **NVIDIA T4 Tensor Core GPUs (Turing architecture, sm_75)**.

---

## 1. AWS Hardware & Architecture Target

| Specification | AWS EC2 g4dn.xlarge Details |
| :--- | :--- |
| **GPU** | 1x NVIDIA T4 Tensor Core (16 GB GDDR6) |
| **Compute Capability** | `sm_75` (Turing) |
| **vCPUs** | 4 Intel Xeon Scalable (Cascade Lake) |
| **System Memory** | 16 GiB RAM |
| **Storage** | 125 GB NVMe SSD |
| **Network** | Up to 25 Gbps |

Nevrine's `CMakeLists.txt` is pre-configured with `CMAKE_CUDA_ARCHITECTURES="75;89"`, compiling native machine code optimized for both the T4 (`sm_75`) and development Ada GPUs (`sm_89`).

---

## 2. Automated Provisioning via Cloud-Init

When launching the EC2 instance in the AWS Console or via AWS CLI, provide [`cloud-init.yaml`](file:///deploy/aws/cloud-init.yaml) in the **User Data** field:

```bash
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxx \
    --user-data file://deploy/aws/cloud-init.yaml \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Nevrine-Inference-Node}]'
```

---

## 3. Running with Docker Compose (Recommended)

1. SSH into the instance:
   ```bash
   ssh -i your-key.pem ubuntu@<INSTANCE-PUBLIC-IP>
   ```

2. Clone and start:
   ```bash
   git clone https://github.com/rahulji0805/Nevrine.git
   cd Nevrine
   docker compose -f docker/docker-compose.yml up -d --build
   ```

3. Verify running container and logs:
   ```bash
   docker ps
   docker logs -f nevrine-server
   ```

---

## 4. Verification & Health Checks

### Check System & GPU Health
```bash
curl -s http://localhost:8000/health | jq .
```
Expected output:
```json
{
  "status": "ok",
  "engine": "Nevrine Neural Acceleration Engine",
  "version": "0.6.0",
  "cuda_device_id": 0,
  "cuda_device_name": "Tesla T4",
  "vram_total_mib": 15360.0,
  "vram_free_mib": 14920.0,
  "vram_used_mib": 440.0
}
```

### Run Real-Time In-Flight Latency Benchmark
```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"iterations": 100, "warmup": 10, "batch_size": 1, "backend": "cuda_tiled"}' | jq .
```
Expected output gives statistical breakdown (`throughput_samples_per_sec`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`).
