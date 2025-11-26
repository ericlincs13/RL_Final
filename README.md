# 筆記

## Training

```bash
cd final_project_env
python client.py --env-random-inverse
```

## Inference

```bash
wsl
docker-compose up -d --build
cd final_project_env
python client.py --eval --url http://localhost:5000 --model ppo_weights/best_model.zip
```