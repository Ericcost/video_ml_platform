# 🏐 Volleyball Analyzer — Niveau 0

Système d'analyse vidéo de volleyball basé sur l'IA.
Lancé entièrement avec **une seule commande**.

---

## 🚀 Démarrage

```bash
# 1. Cloner / se placer dans le dossier
cd volleyball-analyzer

# 2. Lancer tout le système
docker-compose up --build

# 3. Ouvrir l'interface
open http://localhost:8501
```

C'est tout. ✅

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   docker-compose                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Frontend   │    │      API (FastAPI)        │   │
│  │  (Streamlit) │───▶│  POST /upload             │   │
│  │   :8501      │    │  GET  /status/{id}        │   │
│  └──────────────┘    │  GET  /result/{id}        │   │
│                      │  GET  /video/{id}  :8000  │   │
│                      └────────────┬─────────────┘   │
│                                   │ Celery task      │
│                         ┌─────────▼──────────┐      │
│                         │   Worker (Celery)   │      │
│                         │  YOLOv8n detection  │      │
│                         │  IoU tracking       │      │
│                         │  Action heuristics  │      │
│                         └─────────┬──────────┘      │
│              ┌────────────────────┼──────────┐       │
│    ┌─────────▼───┐  ┌────────────▼──┐ ┌──────▼────┐ │
│    │    Redis    │  │   MongoDB    │ │   MinIO   │ │
│    │  (queue)   │  │  (résultats) │ │  (vidéos) │ │
│    └─────────────┘  └──────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 🤖 Pipeline ML

```
Vidéo uploadée
      ↓
[1] YOLOv8n         Détecte ballon (COCO class 32) + joueurs (class 0)
      ↓
[2] Calibration     Auto-détecte les 2 couleurs de maillot dominantes → team_a / team_b
      ↓
[3] IoU Tracker     Suit les objets entre frames, assigne des track_id stables
      ↓
[4] BallTrajectory  Fenêtre glissante de 60 frames : vitesse, direction, hauteur
      ↓
[5] ActionClassifier Règles géométriques → serve / pass / set / attack / block / dig
      ↓
[6] EventSegmenter  Regroupe les frames en événements (min 8 frames stables)
      ↓
Vidéo annotée + JSON de résultats
```

---

## 🎯 Actions détectées

| Action  | Critères heuristiques                                         |
|---------|---------------------------------------------------------------|
| Serve   | Ballon près de la ligne de fond, haute vitesse, trajectoire horizontale |
| Attack  | Vitesse élevée, trajectoire descendante, proche du filet      |
| Block   | Ballon rebondit vers le haut près du filet, bras levés        |
| Set     | Vitesse lente, arc montant, zone médiane du terrain           |
| Dig     | Ballon très bas, rebond ascendant depuis le sol               |
| Pass    | Vitesse modérée, trajectoire montante                         |

---

## 🔌 API REST

| Méthode | Endpoint           | Description                    |
|---------|--------------------|--------------------------------|
| POST    | `/upload`          | Upload vidéo → retourne job_id |
| GET     | `/status/{job_id}` | Statut + progression (0→1)     |
| GET     | `/result/{job_id}` | Résultat JSON complet          |
| GET     | `/video/{job_id}`  | Stream vidéo annotée           |
| GET     | `/health`          | Health check                   |

---

## 📁 Structure du projet

```
volleyball-analyzer/
├── docker-compose.yml
├── api/
│   ├── Dockerfile
│   ├── main.py          ← FastAPI endpoints
│   ├── models.py        ← Pydantic schemas
│   └── requirements.txt
├── worker/
│   ├── Dockerfile
│   ├── tasks.py         ← Celery tasks
│   ├── detector.py      ← YOLOv8 detection + team color
│   ├── tracker.py       ← IoU tracking + ball trajectory
│   ├── action_classifier.py ← Heuristic action classification
│   ├── video_processor.py   ← Orchestration pipeline
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   ├── app.py           ← Streamlit UI
│   └── requirements.txt
└── models/              ← YOLOv8 weights (auto-downloaded at build)
```

---

## ⚡ Performances CPU

Sur MacOS M-series ou Intel i7 :
- Traitement : ~2–5× la durée de la vidéo (ex : vidéo 1min → ~2–5min)
- `frame_skip=2` est activé par défaut (traite 1 frame sur 2, accélère ×2)
- Pour plus de précision : changer `frame_skip=1` dans `worker/tasks.py`
- Pour plus de vitesse : `frame_skip=3` ou `frame_skip=4`

---

## 🔧 Prochaines étapes (Niveau 1)

- Auth par token dans le header HTTP
- Déploiement automatique sur VM
- Tracking usage par utilisateur
