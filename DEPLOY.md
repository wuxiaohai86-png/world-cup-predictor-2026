# 世界杯预测器 — 网页版部署指南

## 方案 A：Streamlit Cloud（推荐，免费）

### 1. 上传到 GitHub
```bash
cd ~/world-cup-predictor-2026
git init
git add .
git commit -m "World Cup 2026 AI Predictor"
git remote add origin https://github.com/YOUR_USERNAME/world-cup-predictor-2026.git
git push -u origin main
```

### 2. 连接 Streamlit Cloud
- 打开 https://share.streamlit.io
- 用 GitHub 账号登录
- 点击 "New app"
- 选择仓库 `world-cup-predictor-2026`
- Main file path: `streamlit_app.py`
- 点击 Deploy

### 3. 获得链接
- 部署完成后获得 URL（如 `https://your-app.streamlit.app`）
- 这个链接可以直接发到闲鱼给买家

---

## 方案 B：Hugging Face Spaces（备选）

1. 打开 https://huggingface.co/spaces
2. Create new Space → Streamlit
3. 上传所有文件
4. 自动部署，获得 `https://huggingface.co/spaces/YOUR_USER/world-cup`

---

## 本地预览
```bash
cd ~/world-cup-predictor-2026
streamlit run streamlit_app.py
```
浏览器打开 http://localhost:8501
