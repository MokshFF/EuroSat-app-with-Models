# 🛰️ EuroSAT Multimodal Land Classification — Web App

**B.Tech 4th Semester | CSET301 AIML | 2025**

Full-stack web application deploying your trained ResNet-18 land classification models with a Flask backend and live inference frontend.

---

## 📁 Project Structure

```
eurosat_app/
│
├── app.py                        ← Flask backend (PyTorch inference API)
├── requirements.txt              ← Python dependencies
├── README.md
│
├── templates/
│   └── index.html                ← Frontend (auto-served by Flask)
│
└── models/                       ← ✅ Your trained weights (already included)
    ├── resnet18_optical_best.pth  (Model A — 44 MB)
    ├── resnet18_sar_best.pth      (Model B — 44 MB)
    └── resnet18_fusion_best.pth   (Model C — 88 MB)
```

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run the server

```bash
python app.py
```

You'll see:
```
[INFO] 🛰️  EuroSAT Backend → http://localhost:5000
[INFO]    Device : cpu   (or cuda if GPU available)
[INFO]    Models : {'optical': True, 'sar': True, 'fusion': True}
```

### Step 3 — Open the app

Visit: **http://localhost:5000**

The status bar at the top will turn **green** when the server is live and all 3 models are detected.

---

## 🔌 API Endpoints

| Method | Endpoint         | Description                             |
|--------|------------------|-----------------------------------------|
| GET    | `/`              | Serves the web frontend                 |
| GET    | `/api/status`    | Server health + model availability      |
| POST   | `/api/classify`  | Classify an uploaded image              |
| GET    | `/api/classes`   | List all 10 class names + descriptions  |

### POST /api/classify — Example

```bash
curl -X POST http://localhost:5000/api/classify \
  -F "image=@my_patch.png" \
  -F "model=fusion"
```

**Response:**
```json
{
  "prediction":    "Forest",
  "confidence":    0.9234,
  "description":   "Dense tree cover",
  "model_type":    "fusion",
  "device":        "cpu",
  "probabilities": {
    "AnnualCrop": 0.0120,
    "Forest":     0.9234,
    "HerbaceousVegetation": 0.0310,
    ...
  }
}
```

---

## 🌐 Deploy Online (Free)

### Option A — Render.com (Recommended)
1. Push this folder to GitHub
2. Go to https://render.com → New Web Service → Connect repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Done — get a public URL like `https://eurosat.onrender.com`

### Option B — Ngrok (Instant demo, no GitHub needed)
```bash
# Terminal 1
python app.py

# Terminal 2
ngrok http 5000
```
Share the ngrok URL with anyone instantly!

---

## 📊 Model Architecture

| Model | Input | Architecture | Params |
|-------|-------|-------------|--------|
| **A — Optical** | RGB (B4,B3,B2) | ResNet-18 → FC(512→256→10) | 11.2M |
| **B — SAR**     | NIR/SWIR (B8,B11,B12) | ResNet-18 → FC(512→256→10) | 11.2M |
| **C — Fusion**  | RGB + NIR/SWIR | Dual ResNet-18 → Concat(1024) → FC | 22.5M |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 5000 busy | Change `port=5000` to `port=5001` in last line of `app.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Status bar shows red | Make sure `python app.py` is running |

---

*Notebook: EuroSat_Main_Ai.ipynb | Backend: app.py | Frontend: templates/index.html*
