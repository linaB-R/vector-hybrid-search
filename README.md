# Búsqueda Híbrida Vectorial

Un **sistema inteligente de búsqueda y coincidencia híbrida** basado en bases de datos vectoriales.  
Este repositorio implementa el prototipo práctico de mi Tesis de Máster:  
*"Sistema Inteligente de Búsqueda y Coincidencia basado en Almacenes Vectoriales" (Universidad Europea de Madrid, 2025).*

El objetivo es combinar **recuperación léxica (BM25 en PostgreSQL)** con **recuperación semántica (embeddings pgvector)** para permitir búsquedas eficientes en catálogos de productos multimodales (texto + imágenes).  
El caso de uso objetivo son pequeñas tiendas de comercio electrónico (ej., productos impresos en 3D) que necesitan búsqueda inteligente escalable y de bajo costo.

---

## 📌 Características Principales
- **PostgreSQL + pgvector** esquema y migraciones.
- **Diseño de búsqueda híbrida**: léxica (BM25) + ANN vectorial (HNSW) + fusión de resultados.
- **Ingesta de datos**: CSV desde S3 público → filtrado → Parquet → Supabase/PostgreSQL.
- **Relleno de embeddings**:
  - Embeddings de texto (Jina v3 / JE-3).
  - Embeddings de imagen (CLIP v2, multimodal).
- **Integración con Supabase**: despliegue fácil vía conexión GitHub.
- **Pipeline extensible** para futuros experimentos de búsqueda híbrida y multimodal.

---

## 🏗️ Fundamentos Tecnológicos y Arquitectura de Datos

### La Importancia Estratégica de PostgreSQL con pgvector

La elección de **PostgreSQL** como sistema de gestión de bases de datos relacionales para este proyecto no es casual, sino que responde a necesidades específicas de escalabilidad, robustez y capacidades avanzadas de indexación que son fundamentales para el éxito de un sistema de búsqueda híbrida.

PostgreSQL se distingue por su **arquitectura extensible** que permite la integración de extensiones especializadas como **pgvector**, transformándolo de un RDBMS tradicional en una plataforma híbrida capaz de manejar tanto datos estructurados relacionales como vectores de alta dimensionalidad. Esta dualidad es crucial para nuestro sistema, ya que permite mantener la **consistencia ACID** de las transacciones relacionales mientras se ejecutan operaciones de búsqueda vectorial de manera nativa y eficiente.

La extensión **pgvector** representa un avance significativo en la democratización de las bases de datos vectoriales. A diferencia de soluciones especializadas como Pinecone, Weaviate o Chroma, pgvector permite implementar capacidades de búsqueda semántica sin la complejidad operacional de mantener sistemas distribuidos separados. Esto resulta especialmente relevante para pequeñas y medianas empresas que requieren capacidades de búsqueda inteligente sin los costos asociados a infraestructuras complejas.

### Índices HNSW: Eficiencia en Búsqueda de Vecinos Más Cercanos

Los **índices Hierarchical Navigable Small World (HNSW)** implementados en pgvector constituyen el núcleo algorítmico que hace viable la búsqueda vectorial a escala. HNSW representa una evolución significativa sobre algoritmos tradicionales como LSH (Locality-Sensitive Hashing) o árboles KD, ofreciendo una **complejidad logarítmica** en las operaciones de búsqueda mientras mantiene alta precisión en la recuperación.

La arquitectura jerárquica de HNSW construye múltiples capas de grafos donde cada nivel superior actúa como un "mapa de carreteras" que guía la navegación hacia regiones prometedoras del espacio vectorial. Esta estructura permite que las consultas de vecinos más cercanos (k-NN) se ejecuten en **tiempo sublineal**, una característica esencial cuando se trabaja con catálogos de productos que pueden contener millones de elementos.

Los parámetros configurables de HNSW (`m`, `ef_construction`, `ef_search`) permiten ajustar el balance entre **precisión, velocidad y uso de memoria**, adaptándose a las características específicas de cada dominio de aplicación. En el contexto de comercio electrónico, donde la latencia de respuesta impacta directamente en la experiencia del usuario, esta flexibilidad es fundamental.

### Búsqueda Híbrida: Sinergia entre Recuperación Léxica y Semántica

La implementación de **búsqueda híbrida** que combina BM25 (Best Matching 25) con embeddings vectoriales representa una aproximación holística al problema de recuperación de información. BM25, basado en el modelo probabilístico de Robertson-Sparck Jones, excele en la coincidencia exacta de términos y manejo de frecuencias, mientras que los embeddings capturan relaciones semánticas latentes que trascienden la coincidencia léxica superficial.

Esta complementariedad es especialmente valiosa en catálogos de productos multilingües o con terminología técnica variada, donde un usuario puede buscar "smartphone resistente al agua" y encontrar productos descritos como "teléfono móvil con certificación IP68". La **fusión de rankings** mediante técnicas como Reciprocal Rank Fusion (RRF) o aprendizaje automático permite combinar las fortalezas de ambos enfoques de manera óptima.

### Amazon S3: Escalabilidad y Economía en Almacenamiento de Contenido Multimedia

La decisión de utilizar **Amazon S3** para el almacenamiento de imágenes responde a consideraciones tanto técnicas como económicas que son críticas para la viabilidad comercial del sistema. S3 ofrece **durabilidad del 99.999999999% (11 9's)** y disponibilidad del 99.99%, garantizando que el contenido multimedia permanezca accesible incluso ante fallos de infraestructura.

La **arquitectura de almacenamiento por objetos** de S3 elimina las limitaciones de sistemas de archivos tradicionales, permitiendo escalabilidad prácticamente ilimitada sin degradación del rendimiento. Esto es fundamental cuando se considera que las imágenes de productos de alta resolución pueden ocupar varios megabytes cada una, y un catálogo empresarial puede contener cientos de miles de productos.

El modelo de **precios por uso** de S3, combinado con opciones de almacenamiento inteligente (S3 Intelligent-Tiering) y clases de almacenamiento de acceso infrecuente, permite optimizar costos automáticamente basándose en patrones de acceso reales. Para startups y PYMEs, esta elasticidad económica es crucial para mantener márgenes sostenibles mientras se escala el negocio.

### Integración Arquitectónica y Consideraciones de Rendimiento

La arquitectura propuesta aprovecha las **características de localidad** inherentes tanto en PostgreSQL como en S3. Las consultas híbridas se ejecutan completamente en PostgreSQL, minimizando la latencia de red, mientras que las imágenes se sirven directamente desde S3 con **CloudFront CDN** para optimizar la entrega global.

Esta separación de responsabilidades permite **escalado independiente** de cada componente: la base de datos puede optimizarse para throughput de consultas mientras que el almacenamiento de imágenes se escala horizontalmente según demanda. La **consistencia eventual** entre metadatos en PostgreSQL e imágenes en S3 se gestiona mediante patrones de sincronización asíncrona que mantienen la integridad del sistema sin bloqueos.

La implementación de **conexiones pooling** y **prepared statements** en PostgreSQL, combinada con **multipart uploads** y **transfer acceleration** en S3, garantiza que el sistema pueda manejar cargas de trabajo concurrentes típicas de aplicaciones de comercio electrónico en producción.

---

## ⚙️ Inicio Rápido

### 1. Aplicar migraciones de base de datos
```powershell
python database\apply_migrations.py
````

### 2. Ingestar dataset desde S3 público → Parquet

```powershell
python src\ingest\ingest_s3_csv.py --max-records 100000 --output-file "data/20250123_data.parquet"
```

### 3. Subir datos filtrados a Supabase/PostgreSQL

```powershell
python .\src\loader\upload_parquet_to_supabase.py
```

### 4. Rellenar embeddings

**Embeddings de texto (JE-v3):**

```powershell
python -m src.ingest.backfill_text_je3 --batch-size 256
```

**Embeddings de imagen (CLIP-v2):**

```powershell
python -m src.ingest.backfill_clip_v2 --mode image --batch-size 64
```

**Opcional – Embeddings de texto CLIP:**

```powershell
python -m src.ingest.backfill_clip_v2 --mode text --batch-size 128
```

---

## 📦 Verificaciones opcionales de AWS CLI

Inspeccionar o descargar imágenes individuales del dataset público:

```powershell
aws s3 ls s3://glovo-products-dataset-d1c9720d/dataset/YKKVTDF_0000672_1629430873.png --no-sign-request 

aws s3 cp s3://glovo-products-dataset-d1c9720d/dataset/YKKVTDF_0000672_1629430873.png . --no-sign-request
```

---

## 🗂️ Estructura del Proyecto

```
lina/
├── README.md
├── requirements.txt
├── changelog/
│   └── CHANGELOG.md
├── data/
│   ├── data.parquet
├── database/
│   ├── apply_migrations.py
│   └── migrations/
│       ├── 00_enable_extensions.sql
│       ├── 01_create_schema.sql
│       ├── 02_tables.sql
│       ├── 03_indexes_hnsw.sql
│       └── [otros archivos SQL]
├── models_cache/
│   ├── models--jinaai--jina-embeddings-v3/
│   └── models--jinaai--xlm-roberta-flash-implementation/
├── sandbox/
│   ├── check_parquet.py
│   └── sample_foodi_dataset.py
├── src/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── clip_v2.py
│   │   └── text_je3.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── backfill_clip_v2.py
│   │   └── backfill_text_je3.py
│   └── loader/
│       └── upload_parquet_to_supabase.py
├── .env.sample   # variables de entorno de ejemplo (añadir aquí los secretos necesarios)
└── venv/         # entorno virtual (excluido en .gitignore)
```

---

## 🔮 Próximos Pasos

* Implementar la interfaz de ejecución de consultas híbridas.
* Evaluar rendimiento con Precision@K, Recall@K, MAP, nDCG.
* Hacer benchmark de parámetros del índice HNSW (`m`, `ef_search`, `ef_construction`).
* Preparar documentación y resultados de la tesis.

---

## 📖 Referencias

* [pgvector](https://github.com/pgvector/pgvector)
* Radford et al. (2021) CLIP: Contrastive Language-Image Pretraining
* Manning et al. (2008) *Introduction to Information Retrieval*
* Jurafsky & Martin (2023) *Speech and Language Processing*

