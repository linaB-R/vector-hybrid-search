### Objetivo General:
Diseñar y validar un motor de recuperación multimodal para datos heterogéneos que combine búsqueda léxica y semántica (texto e imagen) mediante almacenamiento vectorial que integre datos estructurados y representaciones semánticas. La implementación de referencia se realiza en PostgreSQL/pgvector, manteniendo la arquitectura agnóstica a tecnología para facilitar su adopción en otras plataformas equivalentes, con foco en relevancia y eficiencia operativa en pymes.

### Objetivos específicos
1. Diseñar la arquitectura de datos multimodal sobre PostgreSQL/pgvector (esquema, migraciones e índices), manteniendo abstracciones agnósticas para portabilidad a otras plataformas equivalentes.
2. Implementar la ingesta reproducible (S3 → filtrado → Parquet → Postgres) con validaciones por defecto y CLI, asegurando calidad y trazabilidad de datos.
3. Integrar y evaluar embeddings de texto e imagen (JE‑3, E5, GTE, CLIP 1024D/512D); realizar backfill eficiente y crear índices HNSW adecuados por espacio vectorial.
4. Construir el motor de búsqueda híbrida (BM25 + vectores) con fusión de rankings (p. ej., RRF) y recuperación multimodal texto↔imagen.
5. Implementar un RAG mínimo de demostración e integrar LangChain para generar y ejecutar SQL sobre Postgres, mostrando de forma amigable cómo el contexto recuperado sustenta las respuestas y consultas.
6. Definir y ejecutar un plan de evaluación centrado en:
   - Calidad de SQL: Exact Match, Execution Success y correctitud semántica contra gold sets.
   - Métricas IR: Precision@K, Recall@K y nDCG@K sobre consultas de prueba.
   - Factualidad/Faithfulness del RAG: grado en que las respuestas están sustentadas por el contexto recuperado.
   - Costo operativo: estimación por consulta/sesión (tokens LLM + DB).