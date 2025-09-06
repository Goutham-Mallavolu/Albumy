# Albumy (Option B – Semantic Search & Recommendations)

This project extends [Albumy](https://github.com/greyli/albumy) with **ML-powered retrieval features**:

- **Semantic Search** – search photos using natural language, beyond keyword tags.  
- **Find Similar** – retrieve visually/semantically related photos from any photo page.  
- **Configurable Embedding Pipeline** – powered by [SentenceTransformers](https://www.sbert.net/) CLIP embeddings, with fallback logic.  
- **Vector Index** – FAISS (if available) or NumPy cosine similarity.

---

## Setup

### 1. Clone & enter project
```bash
git clone <your-org-repo-url> albumy-optionB
cd albumy-optionB
```

### 2. Create environment (example with conda)
```bash
conda create -n albumy python=3.10 -y
conda activate albumy
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Semantic dependencies
pip install "sentence-transformers==2.2.2" "transformers>=4.38,<5"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu    # optional, speeds up search

# Imaging
pip install Pillow requests
```

---

## Database Init

```bash
flask --app albumy:create_app initdb
flask --app albumy:create_app init   # create admin user
```

---

## Forge fake Data

### Generate demo users, avatars, photos, comments, follows & collects:
```bash
flask --app albumy:create_app forge  
```

---

## Embedding & Index Management

After adding new photos:

1. Compute embeddings:
   ```bash
   flask --app albumy:create_app backfill-embeddings
   ```

2. Build/rebuild the index:
   ```bash
   flask --app albumy:create_app index-embeddings --rebuild
   ```

3. Check count:
   ```bash
   flask --app albumy:create_app shell
   >>> from albumy.models import Embedding
   >>> Embedding.query.count()
   ```

---

## Run the App

```bash
flask --app albumy:create_app run
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Features

- **Search UI**
  - Tabs: **Photo | Semantic | User | Tag**
  - Semantic mode shows *semantic match* badge + similarity score
  - Keyword mode still supported (Whoosh)
- **Find Similar**
  - On photo detail page, click **Find Similar**
- **Telemetry**
  - Logs clicks on “Find Similar” for analysis
- **Config Defaults** (`settings.py`):
  ```python
  SEMANTIC_MODEL_NAME = 'clip-ViT-B-32'
  SEMANTIC_INDEX_PATH = os.path.join(basedir, 'data', 'semantic.index')
  SEMANTIC_MAPPING_PATH = os.path.join(basedir, 'data', 'semantic.ids.npy')
  SEMANTIC_TOPK = 12
  ```

---

## Production Concerns

- **Index updates** – background workers for batch/stream updates  
- **Latency** – FAISS recommended for large datasets  
- **Bias & Explainability** – CLIP may encode biases; results not always explainable  
- **Duplicates** – consider deduplication of near-identical photos  
- **Privacy** – avoid embedding sensitive user captions

---

## Example Workflow

```bash
# Fresh DB + admin
flask initdb
flask init

# Seed demo data
flask --app albumy:create_app forge  

# Build embeddings + index
flask --app albumy:create_app backfill-embeddings
flask --app albumy:create_app index-embeddings --rebuild

# Run
flask --app albumy:create_app run
```

Now visit the site and try:

- Search for **“beach”** in Semantic tab  
- Open a photo → click **Find Similar** Button
