# video-ml-platform

Base minimale d’une plateforme backend pour analyse vidéo (sans ML pour l’instant).

Objectif actuel :
- Repo propre
- Backend Dockerisé
- Service FastAPI fonctionnel
- Endpoint HTTP recevant une requête JSON et validant son schéma

👉 Aucun traitement ML, aucune vidéo, aucune persistance pour l’instant.

---

## 🧱 Architecture



video-ml-platform/
├── docker-compose.yml
└── backend/
├── Dockerfile
├── main.py
└── models.py


---

## 🚀 Stack technique

- Python 3.11
- FastAPI
- Uvicorn
- Docker & Docker Compose

---

## 📦 Fonctionnalités actuelles

### Backend API

- Démarre via Docker
- Exposé sur `http://localhost:8000`
- Endpoint POST `/analyze`
- Validation automatique du schéma JSON via Pydantic

---

## 📡 API

### `POST /analyze`

#### Payload attendu

```json
{
  "video_id": "test",
  "excluded_timeframes": []
}


video_id : string

excluded_timeframes : liste (vide pour l’instant)

Réponse
{
  "status": "received"
}

▶️ Lancer le projet
Prérequis

Docker

Docker Compose (v2)

Vérification :

docker --version
docker compose version

Démarrage

À la racine du repo :

docker compose up --build


Le backend est alors accessible sur :

http://localhost:8000

Test rapide avec curl
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test",
    "excluded_timeframes": []
  }'


Réponse attendue :

{"status":"received"}

🛠️ Commandes utiles

Arrêter les containers :

docker compose down


Relancer avec rebuild :

docker compose up --build


Voir les logs :

docker compose logs -f backend

🧠 Notes de structure (important)

Le backend n’est pas encore un package Python

Les imports sont absolus (ex: from models import AnalyzeRequest)

Le service est lancé via :

uvicorn main:app


👉 Ne pas utiliser d’imports relatifs (from .models ...) dans cette configuration.

🛣️ Prochaines évolutions possibles

Typage strict de excluded_timeframes

Hot reload pour le développement

Structuration en package (app/)

Ajout de traitements asynchrones

Intégration ML / worker séparé

Stockage / file d’attente
