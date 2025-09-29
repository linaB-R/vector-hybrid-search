# RAG Chat Demo — README (MVP)

## Ejecución rápida (API)

```bash
uvicorn src.rag.chat_demo:app --reload --port 8000
```

---

## Objetivo

Levantar un demo **multimodal RAG** que:

* Permita búsqueda **texto ↔ texto** (JE3) y **imagen ↔ texto** (CLIP) sobre **Supabase Postgres + pgvector**.
* Soporte **Text-to-SQL** (solo lectura) para agregaciones en tablas relacionales.
* Use la **Responses API de OpenAI** para la composición final de respuestas y llamadas opcionales a herramientas ([OpenAI][7]).

---

## 1) Estructura del proyecto (archivos RAG nuevos)

```
src/rag/
├── config.py
├── vector_store.py
├── hybrid_retriever.py
├── sql_tools.py
├── graph_workflow.py
└── chat_demo.py
```

> Estos son solo **esqueletos**. Los ingenieros deben completar la implementación.

---

## 2) Requerimientos

En `requirements.txt` incluir:

```
# Núcleo
fastapi
uvicorn

# Entorno y tipos
python-dotenv
pydantic

# Base de datos
psycopg2-binary          
sqlalchemy               

# RAG & Graph
langchain
langchain-openai
langchain-postgres       # integración activa con PGVector (psycopg3)
langgraph                # orquestación de grafos y estados

# (Opcional) Rerankers / extras
numpy
scikit-learn
```

> Nota: `langchain-postgres` es la integración moderna y mantenida para PGVector, reemplaza al wrapper comunitario anterior ([LangChain Docs][1]).

---

## 3) Variables de entorno

Crear/actualizar `.env` con tus credenciales:

```
# Opción 1 (recomendada con Supabase): DSN completo; sslmode se fuerza
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME

# Opción 2: campos discretos (sslmode=require se fuerza en el código)
user=YOUR_DB_USER
password=YOUR_DB_PASSWORD
host=YOUR_DB_HOST
port=5432
dbname=YOUR_DB_NAME

OPENAI_API_KEY=sk-...
```

> Se soporta Supabase Pooler con `DATABASE_URL`. Se usa `sslmode=require` y se ajusta `hnsw.ef_search` en cada query ([Supabase Docs][11]).

---

## 4) Ejecución local (desarrollo)

1. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar acceso a la base de datos**

   ```bash
   python -c "import psycopg2, os; from dotenv import load_dotenv; load_dotenv(); print('Connecting...'); c=psycopg2.connect(user=os.getenv('user'), password=os.getenv('password'), host=os.getenv('host'), port=os.getenv('port'), dbname=os.getenv('dbname')); cur=c.cursor(); cur.execute('select now()'); print(cur.fetchone()); cur.close(); c.close(); print('OK')"
   ```

3. **Levantar la API**

   ```bash
   uvicorn src.rag.chat_demo:app --reload --port 8000
   ```

4. **Probar el endpoint**

   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
           "text": "productos con chocolate sin azúcar",
           "image_b64": null,
           "mode": "auto",
           "k": 20,
           "ef_search": 40
         }'
   ```

### Logging y trazabilidad

Ejecutar todos los tests:

```bash
pytest -q
```

Esto genera logs en `log/chat_run_<timestamp>/`, incluyendo:

* `00_meta.json` → contexto de ejecución y parámetros
* `01_request.json` → request completo
* `02_response_status.json` → código HTTP + latencia
* `03_response_json.json` → respuesta completa
* `04_sql.json` → SQL ejecutado (vectorial + léxico)
* `05_candidates_vector_top.json` → top-5 vectorial
* `06_candidates_lexical_top.json` → top-5 lexical
* `07_final_fused.json` → lista fusionada por RRF

---

## 5) Checklist de pruebas (4 consultas demo)

1. **Texto → Vector (JE3)**
   `productos con chocolate sin azúcar`
   → Deben aparecer ítems relevantes vía JE3 o híbrido.

2. **Imagen → Vector (CLIP)**
   Subir una foto como `image_b64`.
   → Esperar resultados similares vía `clip_image_emb`.

3. **Híbrido (BM25 + Vector + RRF)**
   `galletas integrales sin azúcar marca x`
   → Resultados fusionados (lexical + vectorial) ([Microsoft Learn][5]).

4. **Text-to-SQL (solo lectura)**
   `¿Cuántos productos por colección en Bogotá?`
   → Genera un `SELECT` seguro, limitado, y lo devuelve con tabla.

---

## 6) Pasos para construir el MVP

1. Configuración de entorno (`.env`, `config.py`).
2. Implementar **VectorStoreAdapter** con 3 retrievers (texto JE3, texto CLIP, imagen CLIP).
3. Crear **Hybrid Retriever** (BM25 + vectorial + RRF).
4. Implementar **SQL Tools** (solo SELECT, validación, limit).
5. Definir **Graph Workflow** en `graph_workflow.py`.
6. Exponer **endpoint /chat** en `chat_demo.py`.
7. Probar con las 4 queries de validación.
8. Ajustar `ef_search` y métricas de rendimiento.

---

## 7) Referencias principales

* [LangChain PGVector][1]
* [Crunchy Data — HNSW en Postgres][2]
* [AWS Blog — pgvector en RDS/Aurora][3]
* [pgvector GitHub][4]
* [Jonathan Katz — Hybrid Search][5]
* [LangChain SQL Toolkit][6]
* [OpenAI Responses API][7]
* [LangGraph Tutorial][8]
* [Supabase — HNSW indexes][11]

---

[1]: https://python.langchain.com/docs/integrations/vectorstores/pgvector/?utm_source=chatgpt.com
[2]: https://www.crunchydata.com/blog/hnsw-indexes-with-postgres-and-pgvector?utm_source=chatgpt.com
[3]: https://aws.amazon.com/blogs/database/accelerate-hnsw-indexing-and-searching-with-pgvector-on-amazon-aurora-postgresql-compatible-edition-and-amazon-rds-for-postgresql/?utm_source=chatgpt.com
[4]: https://github.com/pgvector/pgvector?utm_source=chatgpt.com
[5]: https://jkatz.github.io/post/postgres/hybrid-search-postgres-pgvector/?utm_source=chatgpt.com
[6]: https://python.langchain.com/docs/integrations/tools/sql_database/?utm_source=chatgpt.com
[7]: https://platform.openai.com/docs/api-reference/responses/create?utm_source=chatgpt.com
[8]: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/?utm_source=chatgpt.com
[11]: https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes?utm_source=chatgpt.com

