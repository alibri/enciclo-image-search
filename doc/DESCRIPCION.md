# Desarrollo del Sistema de Búsqueda de Proximidad de Imágenes

Voy a desarrollar un sistema de búsqueda de proximidad de imágenes en Python utilizando FastAPI para las APIs y Gradio para la interfaz web. Incluiré un handler de base de datos que pueda funcionar con varias opciones (PostgreSQL, Milvus o Manticore). Se usará un extractor de vectores adecuado según las ventajas y desventajas de cada uno.

El sistema manejará aproximadamente un millón de imágenes en múltiples formatos y almacenará los datos en volúmenes persistentes en Docker. No se incluirán GitHub Actions. 

Te proporcionaré el código completo con los archivos necesarios para levantar los servicios usando Docker Compose.

---

Este sistema permite indexar imágenes mediante vectores de características (embeddings) y realizar búsquedas de imágenes similares (búsqueda por proximidad) de forma eficiente. A continuación, se describen los componentes y funcionalidades solicitados, junto con ejemplos de implementación en **Python 3.12** utilizando **FastAPI** para las APIs y **Gradio** para la interfaz web. Todo el sistema se desplegará con **Docker Compose**, incluyendo la base de datos de vectores y la aplicación web.

## 1. Base de Datos de Vectores

Para almacenar y buscar los vectores de imágenes, el sistema está diseñado para ser compatible con varias bases de datos de vectores: **PostgreSQL (con extensión pgvector)**, **Milvus** y **Manticore Search**. Se implementará un *handler* o capa de abstracción que permita cambiar de una base de datos a otra con mínima modificación de código. 

- **PostgreSQL + pgvector**: Usando la extensión `pgvector`, Postgres puede almacenar vectores (hasta 2000 dimensiones) y realizar búsquedas de similitud (por ejemplo, con métricas coseno o Euclídea) ([Vector Similarity Search with PostgreSQL’s pgvector - A Deep Dive | Severalnines](https://severalnines.com/blog/vector-similarity-search-with-postgresqls-pgvector-a-deep-dive/#:~:text=Support%20for%20various%20vector%20embedding,types)). La ventaja de esta opción es la integración directa con una base de datos relacional convencional, manteniendo propiedades ACID y permitiendo combinar vectores con datos tabulares ([Vector Similarity Search with PostgreSQL’s pgvector - A Deep Dive | Severalnines](https://severalnines.com/blog/vector-similarity-search-with-postgresqls-pgvector-a-deep-dive/#:~:text=pgvector%20is%20an%20open,the%20rest%20of%20your%20data)). Sin embargo, para volúmenes muy grandes de vectores (millones de imágenes), la búsqueda podría ser más lenta que en bases de datos especializadas a menos que se utilicen índices aproximados (IVFFlat) ofrecidos por pgvector. 

- **Milvus**: Es una base de datos de vectores de alto rendimiento diseñada específicamente para **búsqueda en conjuntos masivos de vectores**. Milvus puede escalar a decenas de **miles de millones** de vectores con mínima pérdida de rendimiento ([Milvus | High-Performance Vector Database Built for Scale](https://milvus.io/#:~:text=Milvus%20is%20an%20open,vectors%20with%20minimal%20performance%20loss)), ofreciendo búsqueda aproximada muy rápida (usando índices como HNSW, IVF, etc.) y soportando múltiples métricas de similitud. La desventaja es que es un sistema independiente que añade complejidad en la infraestructura (requiere despliegue adicional, memoria, etc.), pero para aproximadamente 1 millón de imágenes es una solución robusta (Milvus en modo *standalone* está pensado para hasta millones de vectores en un solo nodo ([Milvus | High-Performance Vector Database Built for Scale](https://milvus.io/#:~:text=%2A%20))). 

- **Manticore Search**: Es un motor de búsqueda que ha incorporado soporte para vectores (*vector search*). Permite almacenar embeddings en campos tipo vector y realizar consultas *k-NN* usando SQL o una API JSON. Sus ventajas incluyen ser una solución abierta y ligera, con soporte nativo de búsqueda vectorial y escalabilidad para conjuntos extensos de imágenes ([Image-to-Image Search with Manticore Search](https://manticoresearch.com/use-case/image-to-image-search/#:~:text=Why%20Manticore%20Search%20is%20good,Image%20Search)). Además, se puede combinar con búsquedas de texto tradicional en caso de necesitar un enfoque híbrido. Como contra, su comunidad es más pequeña que la de Milvus y puede requerir ajuste fino en conjuntos **muy** grandes para rendimiento óptimo ([Image-to-Image Search with Manticore Search](https://manticoresearch.com/use-case/image-to-image-search/#:~:text=Image%3A%20Manticore%20Search%20Logo%20Cons)). 

**Implementación del handler de base de datos**: Abstraeremos las operaciones básicas en una clase o módulo que ofrezca métodos para **almacenar** un vector y **buscar** vectores similares, independientemente del motor subyacente. Por ejemplo, podríamos definir una interfaz sencilla:

```python
# file: db_handler.py
from typing import List

class VectorDBHandler:
    def __init__(self, db_type: str, config: dict):
        self.db_type = db_type
        # Inicializar conexión según tipo de base de datos
        if db_type == "postgres":
            import psycopg2
            self.conn = psycopg2.connect(**config)
        elif db_type == "milvus":
            from pymilvus import connections, Collection
            connections.connect(**config)
            self.collection = Collection(name=config["collection_name"])
        elif db_type == "manticore":
            import pymysql
            self.conn = pymysql.connect(**config)
        else:
            raise ValueError("Tipo de base de datos no soportado")
    
    def add_vector(self, vector: List[float], image_id: str):
        """Almacena un vector y su ID de imagen en la base de datos."""
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            # Suponiendo tabla 'images' con columnas id (texto) y embedding (vector)
            cur.execute("INSERT INTO images (id, embedding) VALUES (%s, %s)",
                        (image_id, vector))
            self.conn.commit()
        elif self.db_type == "milvus":
            # En Milvus, insertamos en la collection
            self.collection.insert([[image_id], [vector]])
        elif self.db_type == "manticore":
            cur = self.conn.cursor()
            # Convertir la lista de floats a formato vector literal de Manticore
            vector_str = ",".join(f"{x:.6f}" for x in vector)
            cur.execute(f"INSERT INTO images (id, embedding) VALUES (%s, VECTOR(%s))",
                        (image_id, vector_str))
            self.conn.commit()
    
    def query_vector(self, vector: List[float], top_k: int = 5) -> List[str]:
        """Busca los vectores más similares al vector dado. Retorna una lista de IDs de imágenes similares."""
        results = []
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            # Usamos el operador <-> de pgvector para calcular distancia (ej. coseno o Euclidea según configuración de la columna)
            cur.execute("SELECT id FROM images ORDER BY embedding <-> %s LIMIT %s;", (vector, top_k))
            results = [row[0] for row in cur.fetchall()]
        elif self.db_type == "milvus":
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            res = self.collection.search([vector], "embedding", params=search_params, limit=top_k)
            # Extraer IDs de resultado
            results = [hit.entity.get("id") for hit in res[0]]
        elif self.db_type == "manticore":
            cur = self.conn.cursor()
            vector_str = ",".join(f"{x:.6f}" for x in vector)
            # Consulta k-NN en Manticore usando ORDER BY <=> (distancia euclídea/coseno según configuración del índice)
            cur.execute(f"SELECT id FROM images ORDER BY embedding <=> VECTOR({vector_str}) LIMIT {top_k};")
            results = [row[0] for row in cur.fetchall()]
        return results
```

En este código de ejemplo, el *handler* `VectorDBHandler` soporta las tres opciones. Observaciones:

- **PostgreSQL**: Se asume que la tabla `images` tiene una columna `embedding` de tipo `VECTOR` (definido por pgvector). La consulta utiliza `ORDER BY embedding <-> %s` donde `<->` es el operador de distancia (por defecto, distancia Euclídea o Coseno según se haya definido el índice) y el parámetro `%s` es el vector de consulta. Se deben instalar y habilitar la extensión pgvector (`CREATE EXTENSION vector;`) en la base de datos Postgres antes de ejecutar estas operaciones.

- **Milvus**: Se usa el cliente `pymilvus`. Antes de insertar o buscar, hay que tener creada una *collection* en Milvus con un campo vectorial (por ejemplo, `embedding`) y quizás un campo de ID. En el ejemplo, se inserta proporcionando listas de IDs y vectores. La búsqueda utiliza `collection.search` con la métrica de distancia configurada (ej. L2). *Nota:* En Milvus la configuración de métricas y tipos de índice se realiza al crear la colección e indexar los datos, fuera de este snippet.

- **Manticore**: Utilizamos `pymysql` para conectarnos via SQL, ya que Manticore expone un protocolo MySQL/SQL. La inserción y búsqueda utilizan sintaxis SQL de Manticore:
  - Al insertar: `VECTOR(<values>)` convierte una lista de números a su tipo vector interno. En la creación de la tabla en Manticore, se define algo como `embedding VECTOR(DIM)`, donde DIM es la dimensión del vector, y en la cláusula `WITH` del índice se especifica la métrica (cosine, l2, etc.).
  - Para buscar, se puede usar `ORDER BY embedding <=> VECTOR(...)` donde `<=>` es el operador de distancia k-NN en Manticore (similar al de pgvector). Esta consulta retorna los IDs de las imágenes más cercanas.

**Nota:** En un entorno real, habría que manejar las conexiones de forma segura (usando pooling, manejo de errores, etc.). Además, para rendimiento en grandes volúmenes, se usarían índices aproximados:
- En Postgres pgvector: crear un índice IVF (ej. `CREATE INDEX ON images USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);`).
- En Milvus: construir índices HNSW, IVF_PQ, etc., después de insertar los datos.
- En Manticore: definir `TYPE = 'columnar'` para almacenamiento y ajustar `INDEX RTINDEX` con parámetros de vector.

## 2. Extracción de Vectores (Feature Extraction)

Para representar las imágenes como vectores numéricos, utilizaremos un **modelo de extracción de características** (*encoder* visual). Se han considerado varios modelos y arquitecturas, cada una con ventajas y desventajas:

- **OpenAI CLIP (ViT o ResNet50)**: CLIP es un modelo multimodal entrenado con 400 millones de pares imagen-texto. Su encoder de imágenes genera embeddings de alta dimensionalidad (ej. 512 dimensiones) con significado semántico, es decir, imágenes *visualmente* o *conceptualmente* similares tienden a tener vectores cercanos. **Ventajas**: ofrece excelentes resultados en comparación semántica, permite incluso comparar imágenes con texto (no requerido aquí, pero denota la riqueza del embedding). Es moderno y alcanza alta precisión en *retrieval*. **Desventajas**: el modelo es grande; requiere más recursos (GPU recomendada para extracción rápida) y la inferencia puede ser más lenta que modelos más pequeños. También, al ser multimodal, puede que muy pequeños detalles visuales no se capturen si no eran relevantes para la correspondencia con texto durante su entrenamiento.

- **ResNet (ej: ResNet50)**: Red neuronal convolucional clásica pre-entrenada en ImageNet para clasificación. Podemos usar la activación de una capa intermedia/penúltima (por ejemplo, la capa *pooling* de 2048 valores) como vector de características. **Ventajas**: modelos disponibles ampliamente, relativamente más ligeros que CLIP, y capturan bien las características visuales básicas (texturas, formas) útiles para medir similitud visual. Pueden funcionar razonablemente para buscar imágenes similares en cuanto a contenido general (p.ej., clasificadas en la misma categoría). **Desventajas**: los embeddings de un modelo puramente de clasificación pueden ser menos semánticos; dos imágenes de la misma clase serán cercanas, pero para similitud *general* (por ejemplo, fotos de estilos o contextos diferentes) podría no ser tan sensible. Además, aunque ResNet50 no es pequeño, existe desde hace tiempo y su rendimiento es menor comparado con arquitecturas más nuevas.

- **VGG**: Una arquitectura más antigua (VGG16/19) que también puede generar vectores (usando sus capas FC de 4096 dimensiones, por ejemplo). **Ventajas**: en su momento fueron usadas para *image retrieval* y aún pueden servir; fáciles de implementar con frameworks actuales. **Desventajas**: muy grandes en tamaño de parámetros, vectores de dimensionalidad alta (4096), rendimiento inferior a modelos más nuevos en cuanto a calidad de embedding, y mucho más lentos en inferencia. Probablemente no sea ideal para un sistema moderno, debido a su ineficiencia.

**Comparación y elección**: Considerando lo anterior, **CLIP** suele ser la mejor opción por la calidad de sus embeddings que encapsulan similitud visual y conceptual. Dado que el sistema puede requerir precisión en encontrar imágenes *parecidas* (incluso si no son idénticas en pixel a pixel), CLIP ofrece mejores resultados (por ejemplo, encuentra imágenes con el mismo objeto o escena aunque difieran en color o ángulo). Por otro lado, si el principal criterio fuera rapidez y simplicidad, un **ResNet50** con capas congeladas podría usarse y probablemente rendir más rápido en CPU. 

En este diseño optaremos por **integrar CLIP** como modelo de extracción de vectores, asumiendo disponibilidad de GPU para acelerar (aunque también puede correrse en CPU más lentamente). Usaremos la implementación abierta de CLIP. Por ejemplo, con la librería `openai/CLIP` o vía HuggingFace:

```python
# file: model.py
import torch
from PIL import Image
import numpy as np
# Usamos el paquete openai/clip o huggingface transformers
import clip  # requiere pip install openai_clip (o similar)

# Cargar el modelo CLIP y la transformación de imágenes
model, preprocess = clip.load("ViT-B/32")  # modelo ViT-B/32 de CLIP
model.eval()  # Modo evaluación, no entrenamiento

def extract_vector(image_bytes: bytes) -> np.ndarray:
    """Recibe una imagen en bytes y devuelve su embedding de características como vector numpy."""
    # Abrir la imagen con PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Preprocesar (redimensionar, centrar, normalizar según CLIP)
    img_t = preprocess(image).unsqueeze(0)  # tensor 1xCxHxW
    with torch.no_grad():
        vector = model.encode_image(img_t)
    # Normalizar el vector (opcional, si queremos comparar por coseno directamente)
    vector_n = vector / vector.norm(dim=-1, keepdim=True)
    embedding = vector_n.cpu().numpy().flatten()
    return embedding
```

En este código:
- Cargamos el modelo CLIP pre-entrenado `ViT-B/32` y su preprocesador asociado. 
- `extract_vector` toma la imagen (como bytes) y devuelve un `np.ndarray` unidimensional (embedding de la imagen). Convertimos a RGB para asegurar compatibilidad con el modelo. 
- Usamos `torch.no_grad()` para no calcular gradientes (ya que solo inferencia). 
- Opcionalmente normalizamos el vector resultante para que tenga norma 1 (útil si usamos distancia coseno en la búsqueda de similitud).
- El resultado `embedding` será un vector de 512 dimensiones (para ViT-B/32).

**Nota:** Si se optara por ResNet50 o VGG, la función sería similar: cargar el modelo de `torchvision.models` preentrenado, remover la capa final de clasificación, ingresar la imagen y extraer el tensor resultante. CLIP ya proporciona directamente la función `encode_image`. En cualquier caso, este módulo de extracción se integrará con la API: es decir, la API recibirá una imagen, llamará a `extract_vector` (o modelo equivalente) y obtendrá el vector para luego almacenarlo o buscar similitudes.

## 3. Manejo de Imágenes (Formatos y Volumen)

El sistema debe aceptar múltiples formatos de imagen comunes como **JPEG, PNG, BMP, GIF**, etc. Para lograr esto, usaremos bibliotecas estándar (Pillow, OpenCV). En el ejemplo anterior usamos **Pillow (PIL)**, que soporta JPEG, PNG y otros formatos por defecto. FastAPI facilita recibir archivos de imagen en la API a través de `UploadFile`, que podemos leer en bytes y procesar con PIL/OpenCV sin importar el formato (siempre que esté entre los soportados, que son la mayoría). 

Algunos detalles a tener en cuenta:
- **Tamaño y preprocesamiento**: Conviene redimensionar/normalizar las imágenes de forma consistente antes de extraer el vector. Por ejemplo, CLIP espera 224x224 píxeles con normalización específica. La función `preprocess` de CLIP se encarga de esto. Para otros modelos, deberíamos manualmente redimensionar la imagen al tamaño de entrada (p.ej. 224x224 para ResNet/VGG) y normalizar canales.
- **Imagen a vector**: Asegurarse de convertir siempre a colores (RGB) y manejar imágenes con canal alfa si son PNG (por eso usamos `.convert("RGB")` para eliminar alfa).

**Escalabilidad a ~1 millón de imágenes**: Un millón de imágenes es un volumen grande pero manejable con la infraestructura adecuada:
- No guardaremos las imágenes en memoria dentro de la aplicación. En su lugar, **almacenamos las imágenes en disco** (por ejemplo, en un volumen montado en Docker) o en un servicio de almacenamiento de objetos (no solicitado explícitamente aquí, así que asumiremos almacenamiento local). Solo se cargarán en memoria al momento de procesarlas para extracción o al consultarlas para mostrar resultados.
- Los **vectores** resultantes (digamos de 512 dimensiones float32) ocupan alrededor de 2 KB cada uno. Un millón de ellos serían ~2 GB, que es posible de almacenar en la base de datos de vectores. Tanto PostgreSQL como Milvus o Manticore pueden manejarlo:
  - En PostgreSQL, 1 millón de filas con un vector de 512 floats cada una es factible, aunque habría que añadir índices *approximate* y quizá particionar si el hardware es limitado.
  - Milvus está especialmente diseñado para millones de vectores, y maneja índices en memoria/SSD eficientemente.
  - Manticore igualmente está pensado para alta velocidad con índices en memoria; 1 millón no es problema, pero se debe configurar el índice vectorial apropiadamente.
- **Inserción masiva**: Para indexar un millón de imágenes inicialmente, se recomienda hacerlo en lotes (batching) en lugar de insertar de a una:
  - En PostgreSQL, usar COPY o batch inserts si es posible.
  - En Milvus, usar inserciones por lotes (pymilvus permite insertar múltiples vectores a la vez).
  - En Manticore, también es posible usar su API bulk o realizar múltiples inserts en transacciones.
- **Búsqueda eficiente**: Con 1e6 vectores, la búsqueda exacta lineal podría tardar demasiado. Es casi obligatorio usar una técnica de **búsqueda aproximada** (ANN):
  - pgvector: índice IVFFlat o HNSW (pgvector actualmente soporta IVFFlat).
  - Milvus: elegir index tipo IVF_FLAT, IVF_PQ, HNSW, etc., y ajustar parámetros (nlist, nprobe).
  - Manticore: su índice columnar con HNSW internamente (creo que Manticore usa HNSW para vectores) y se puede ajustar M y ef.
- **Manejo de resultados**: Cuando se encuentran imágenes similares, probablemente querremos mostrar las imágenes. Para ello, debemos tener alguna forma de mapear el ID o referencia almacenada a la imagen real. Estrategias:
  - *Almacenar meta-datos de la imagen en la BD*: Por ejemplo, nombre de archivo, ruta o URL. En PostgreSQL o Manticore, podemos tener columnas adicionales (ej. `filepath TEXT`). En Milvus, se puede usar el ID como referencia que luego relacionamos con un registro externo. También Milvus 2.0 soporta almacenar data en *fields* no vectoriales, así podríamos almacenar la ruta como un campo.
  - *Convención de nombres*: Podemos usar el mismo ID (ej. un hash o un UUID) como nombre de archivo de la imagen en disco. Así, dado un ID recuperado de la búsqueda, sabemos que la imagen está en `images/<id>.jpg` por ejemplo. Esto simplifica no tener que consultar otra base para obtener la ruta.
  - *Servicio de archivos*: La aplicación FastAPI podría exponer un endpoint para servir la imagen por ID (leyendo del disco) al cliente, o la interfaz web puede acceder directamente al sistema de archivos compartido si está montado.

Resumiendo, el sistema soportará gran cantidad de imágenes gracias al uso de bases de datos especializadas de vectores y técnicas de indexación, y maneja múltiples formatos asegurando la conversión adecuada antes de la extracción del vector.

## 4. APIs de Indexación y Búsqueda

Se implementarán dos endpoints principales en una aplicación **FastAPI**: uno para **almacenar (indexar) imágenes**, y otro para **buscar imágenes similares**. También podemos incluir un tercero para obtener una imagen por ID si es necesario para la interfaz.

**1. API de carga/almacenamiento de vectores** (`POST /index_image`):  
Este endpoint recibe una imagen y la agrega al índice de búsqueda. Pasos del endpoint:
- Recibir el archivo de imagen (por ejemplo, usando `UploadFile` de FastAPI en el formulario).
- Leer los bytes y extraer el vector de características usando el modelo elegido (CLIP en nuestro caso).
- Generar o recibir un identificador único para la imagen:
  - Si la imagen ya tiene algún ID (por ejemplo, en su nombre de archivo), podemos usarlo. Si no, generar uno (un UUID o un hash MD5/SHA del contenido, o un contador autoincremental).
- Almacenar el vector en la base de datos de vectores, mediante el *handler* correspondiente. Guardar también la referencia de la imagen:
  - Opcional: guardar la imagen en disco (si no se hizo previamente) en una carpeta designada, nombrándola con el ID.
  - En la base de datos, almacenar el ID y el vector, y posiblemente la ruta/metadata.
- Retornar una respuesta de éxito (por ejemplo, JSON con el `id` asignado).

**2. API de búsqueda de imágenes similares** (`POST /search_image`):  
Este endpoint recibe una imagen de consulta y retorna las imágenes (o IDs) más similares de la colección indexada. Pasos:
- Recibir la imagen de consulta (de nuevo via `UploadFile`).
- Extraer su vector de características con el mismo modelo.
- Usar el *handler* de base de datos para buscar los *top K* vectores más cercanos en el conjunto (por ejemplo los 5 más similares).
- Obtener los IDs resultantes y, a partir de ellos, las referencias a las imágenes almacenadas (p. ej. nombres de archivo).
- Devolver la lista de resultados. El formato puede ser un JSON con una lista de objetos, cada uno con el `id` y quizás una URL o ruta para obtener la imagen:
  - Por simplicidad, podríamos devolver solo la lista de IDs y luego la interfaz cliente los muestra obteniendo cada imagen por otro endpoint o construyendo la ruta. 
  - O bien, incluir directamente en la respuesta datos para mostrar (por ejemplo, si la interfaz es web, podría aceptar base64 de miniaturas, pero eso puede ser pesado para varias imágenes; mejor solo ID o paths).

A modo de ejemplo, definamos estos endpoints en FastAPI:

```python
# file: main.py
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from model import extract_vector
from db_handler import VectorDBHandler

app = FastAPI(title="Image Similarity Search API")

# Inicializar el handler de la BD de vectores, escogiendo el tipo (por ejemplo, usando env var)
import os
db_type = os.getenv("VECTOR_DB", "postgres")  # elegir "postgres", "milvus" o "manticore"
# Configuraciones de conexión (en un entorno real vendrían de variables de entorno o config file)
if db_type == "postgres":
    db_config = {
        "host": os.getenv("PG_HOST", "db"),  # en Docker, servicio "db"
        "port": 5432,
        "user": os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD", "postgres"),
        "dbname": os.getenv("PG_DATABASE", "images_db")
    }
elif db_type == "milvus":
    db_config = {
        "uri": os.getenv("MILVUS_HOST", "tcp://milvus:19530"),  # ejemplo URI Milvus
        "collection_name": os.getenv("MILVUS_COLLECTION", "images")
    }
elif db_type == "manticore":
    db_config = {
        "host": os.getenv("MANTICORE_HOST", "db"),  # asumiendo 'db' es el servicio de Manticore
        "port": 9306,  # Puerto SQL de Manticore
        "user": os.getenv("MANTICORE_USER", None),
        "password": os.getenv("MANTICORE_PASS", None),
        "database": os.getenv("MANTICORE_INDEX", "")  # En Manticore se selecciona índice tras conectar
    }
else:
    raise RuntimeError("Unsupported DB type")
vector_db = VectorDBHandler(db_type, db_config)

# Carpeta local para guardar imágenes
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "/app/images")

@app.post("/index_image")
async def index_image(file: UploadFile = File(...)):
    # Leer la imagen enviada
    image_bytes = await file.read()
    # Generar ID único (usamos nombre de archivo proporcionado o un UUID)
    import uuid
    image_id = file.filename or str(uuid.uuid4())
    # Extraer el vector de características
    vector = extract_vector(image_bytes)
    # Almacenar la imagen físicamente (opcional, en carpeta compartida)
    image_path = f"{IMAGE_FOLDER}/{image_id}"
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    # Almacenar vector e ID en la base de datos de vectores
    vector_db.add_vector(vector.tolist(), image_id)
    return {"status": "ok", "id": image_id}

@app.post("/search_image")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    image_bytes = await file.read()
    query_vec = extract_vector(image_bytes)
    # Buscar vectores similares en la BD
    similar_ids = vector_db.query_vector(query_vec.tolist(), top_k=top_k)
    # Preparar respuesta con IDs (y potencialmente URLs o rutas)
    results = []
    for sim_id in similar_ids:
        results.append({"id": sim_id, "image_url": f"/image/{sim_id}"})
    return {"query_id": None, "results": results}

@app.get("/image/{image_id}")
def get_image(image_id: str):
    # Devuelve la imagen del directorio dado su ID
    image_path = f"{IMAGE_FOLDER}/{image_id}"
    return FileResponse(image_path)
```

**Explicación del API**:
- Usamos `UploadFile` para recibir archivos binarios en `/index_image` y `/search_image`. Esto permite que un usuario envíe, por ejemplo, un formulario `multipart/form-data` con el campo "file".
- En `/index_image`: 
  - Leemos los bytes completos (`await file.read()`). 
  - Creamos un `image_id`. Aquí usamos `file.filename` (nombre original) si existe, de lo contrario generamos un UUID. (En un sistema real, quizá siempre usaríamos un UUID o un hash para evitar colisiones de nombre).
  - Llamamos `extract_vector` (del módulo de modelo) para obtener el vector de la imagen.
  - Guardamos la imagen en disco (`IMAGE_FOLDER`) con su `image_id` como nombre. Esto facilita luego recuperarla para mostrar. (Asumimos que `IMAGE_FOLDER` existe y está montado en el contenedor).
  - Llamamos `vector_db.add_vector(vector, image_id)` para insertar en la base de datos de vectores. Convertimos a lista normal con `.tolist()` porque, por ejemplo, psycopg2 puede aceptar listas para insertar en columnas de tipo vector, y otras librerías similares. 
  - Devolvemos un JSON sencillo con estatus OK y el `id` asignado.
- En `/search_image`: 
  - Leemos la imagen de consulta en bytes.
  - Extraemos su vector.
  - Usamos `vector_db.query_vector(query_vec, top_k)` para obtener los IDs más similares.
  - Construimos una lista de resultados. Aquí optamos por proporcionar, para cada ID, una URL donde obtener la imagen (`/image/{id}`). Esto implica que implementamos el endpoint `/image/{image_id}` para servirlas.
  - Devolvemos un JSON con la lista de resultados. También incluimos `query_id: None` (podría ser útil si indexáramos la imagen de consulta también, pero en este caso no la guardamos, solo la usamos para búsqueda).
- En `/image/{image_id}`:
  - Utilizamos `FileResponse` de FastAPI para devolver el archivo de imagen directamente dado su ID (buscándolo en `IMAGE_FOLDER`). Esto nos permite que la interfaz web pueda mostrar la imagen solicitando a este URL.

**Detalles**:
- El handler `VectorDBHandler` se inicializa en el arranque de la app según la variable de entorno `VECTOR_DB` (por defecto "postgres" en este ejemplo). En Docker Compose podremos cambiar esa variable para usar Milvus o Manticore.
- `IMAGE_FOLDER` también viene de env var, se supone montada en Docker. En desarrollo local podría ser un path local.
- Para Manticore, note que tras conectar quizás se deba seleccionar la base de datos o índice (`USE <index>`). En el código, `database` en config se puso, pero según instalación podría omitirse. En cualquier caso, el ejemplo asume que la tabla `images` existe con la columna `embedding` vectorial.
- Todos los endpoints manejan potenciales excepciones (no se ve en snippet por brevedad). En producción, habría `try/except` para errores de BD o de modelo, retornando códigos 500 apropiados o mensajes de error.

## 5. Automatización de Procesos

Para facilitar la utilización del sistema, se desarrollarán dos **scripts** auxiliares en Python que consumen las APIs anteriores, automatizando tareas comunes:

**a) Script de indexación masiva**: recorre una carpeta con imágenes y envía cada imagen a la API de indexación (`/index_image`). Esto permite poblar la base de datos de vectores de forma automática. Podría llamarse, por ejemplo, `bulk_index.py`. Su funcionamiento:
- Leer la ruta de la carpeta de imágenes (podría usar `sys.argv` o variables de entorno para la ruta).
- Opcionalmente, filtrar por extensiones de imagen conocidas (.jpg, .png, etc.).
- Iterar por cada archivo de la carpeta (y subcarpetas si se desea).
- Por cada archivo, realizar una petición HTTP POST al endpoint `/index_image` enviando el archivo.
- Manejar la respuesta (p. ej. imprimir confirmación o almacenar los IDs retornados).
- Se puede hacer de forma secuencial o en paralelo (para acelerar, aunque cuidado con sobrecargar el servidor; un grado de concurrencia moderado puede estar bien).

Por ejemplo, usando la librería `requests` en Python:

```python
# file: scripts/bulk_index.py
import os, sys
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/index_image")

def index_folder(folder_path: str):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                file_path = os.path.join(root, filename)
                with open(file_path, "rb") as img:
                    files = {"file": (filename, img, "application/octet-stream")}
                    try:
                        resp = requests.post(API_URL, files=files)
                        resp.raise_for_status()
                        data = resp.json()
                        print(f"Indexed {filename} -> ID: {data.get('id')}")
                    except Exception as e:
                        print(f"Failed to index {filename}: {e}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./images_to_index"
    index_folder(folder)
```

En este script:
- `API_URL` por defecto apunta a `localhost:8000/index_image` (asumiendo que el FastAPI corre local o en ese puerto accesible; en Docker Compose podría ser el nombre del servicio).
- Recorre recursivamente (`os.walk`) la carpeta `images_to_index` (o la proporcionada por argumento).
- Filtra por extensiones de imagen conocidas.
- Por cada imagen, abre el archivo en modo binario y hace `requests.post` al endpoint, enviando el archivo con el campo "file". (Le pasamos también el nombre original en la tupla, aunque no es estrictamente necesario).
- Comprueba si la respuesta es exitosa y muestra el ID asignado. Si hay algún error (excepción en la solicitud HTTP), la captura e imprime un mensaje, continuando con la siguiente imagen.

Esto permitirá, por ejemplo, indexar miles de imágenes automáticamente en la base de datos de vectores. Para ~1 millón de imágenes, se podría dividir en procesos paralelos o asegurarse de que el servidor y la red soporten el volumen de datos.

**b) Script de búsqueda por imagen**: toma una imagen de entrada (ruta en disco) y la envía a la API de búsqueda (`/search_image`), luego procesa la respuesta para mostrar los resultados. Llamémoslo `find_similar.py`. Pasos:
- Cargar la imagen de consulta desde un archivo (ruta proporcionada).
- Hacer una petición POST a `/search_image` con el archivo.
- Recibir la respuesta JSON con los resultados (IDs o URLs).
- Mostrar los IDs de resultados y/o abrir las imágenes correspondientes. En una versión simple de consola, podemos imprimir los IDs. Si queremos ver las imágenes, podríamos descargar cada imagen desde `/image/{id}` o abrir el archivo local si compartimos almacenamiento.

Ejemplo de implementación:

```python
# file: scripts/find_similar.py
import sys, requests, webbrowser

SEARCH_URL = "http://localhost:8000/search_image"
GET_IMAGE_URL = "http://localhost:8000/image"  # base URL to fetch images by ID

def find_similar(image_path: str):
    with open(image_path, "rb") as img:
        files = {"file": (image_path, img, "application/octet-stream")}
        resp = requests.post(SEARCH_URL, files=files)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        print(f"Found {len(results)} similar images for {image_path}:")
        for res in results:
            image_id = res["id"]
            print(f" - ID: {image_id}")
            # Abrir la imagen en el navegador predeterminado, si se desea
            url = f"{GET_IMAGE_URL}/{image_id}"
            webbrowser.open(url)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_similar.py <image_path>")
        sys.exit(1)
    query_image = sys.argv[1]
    find_similar(query_image)
```

Explicación:
- Envía la imagen a buscar de la misma manera que antes (POST con file).
- Extrae la lista de resultados del JSON.
- Imprime cuántas imágenes similares encontró y sus IDs.
- Opcionalmente, utiliza `webbrowser.open` para abrir cada imagen en el navegador usando el endpoint `/image/{id}`. Esto asume que la API es accesible en local; si estuviera desplegado en otro host, se debería ajustar la URL base.

Este script facilita pruebas rápidas: el usuario puede seleccionar una imagen de su sistema de archivos y el script abrirá en el navegador las imágenes similares encontradas por el sistema.

## 6. Interfaz Web con Gradio

Como interfaz web básica, utilizaremos **Gradio** para permitir a los usuarios subir una imagen y visualizar las imágenes similares encontradas. Gradio permite crear rápidamente una aplicación web interactiva sin necesidad de desarrollar HTML/JS desde cero.

**Diseño de la interfaz**:  
Tendremos un componente de entrada *Image* donde el usuario carga o arrastra una imagen de consulta, y un componente de salida que muestre una galería de imágenes similares. Bajo el capó, la interfaz llamará a la API de búsqueda o usará directamente la función de búsqueda.

Podemos integrar Gradio de dos formas:
- Llamando internamente a las funciones de extracción y búsqueda (usando el mismo modelo y handler ya cargados en la app). Esto requeriría ejecutar Gradio dentro del mismo proceso que FastAPI o compartir el código.
- Llamando a la API REST FastAPI como un cliente HTTP normal desde Gradio. Esto desacopla las capas y es sencillo de implementar.

Para simplicidad, usaremos la **segunda opción**: el evento de Gradio hará una petición HTTP a nuestro endpoint `/search_image` y luego recopilará las imágenes resultantes.

Por ejemplo, un sencillo script de Gradio:

```python
# file: app_gradio.py
import gradio as gr
import requests

API_SEARCH_URL = "http://localhost:8000/search_image"

def query_similar_images(input_image):
    # input_image viene como ruta temporal o PIL Image via gradio, lo convertimos si es PIL
    if isinstance(input_image, str):
        # If a filepath string is provided
        img_data = open(input_image, "rb").read()
    else:
        # If a PIL Image or numpy array is provided by Gradio
        import numpy as np
        import io
        from PIL import Image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
        buffered = io.BytesIO()
        input_image.save(buffered, format="PNG")
        img_data = buffered.getvalue()
    # Llamar a la API de búsqueda
    files = {"file": ("query.png", img_data, "application/octet-stream")}
    response = requests.post(API_SEARCH_URL, files=files)
    response.raise_for_status()
    results = response.json().get("results", [])
    # Preparar lista de imágenes para mostrar
    sim_images = []
    for item in results:
        image_id = item["id"]
        # Obtener la imagen desde el endpoint de imagen
        img_resp = requests.get(f"http://localhost:8000/image/{image_id}")
        if img_resp.status_code == 200:
            # Guardar el contenido en un archivo temporal para que Gradio pueda leerlo (o pasar como PIL)
            sim_images.append(Image.open(io.BytesIO(img_resp.content)))
    return sim_images

# Construir la interfaz
title = "Búsqueda de Imágenes Similares"
description = "Sube una imagen y el sistema encontrará imágenes visualmente similares."
interface = gr.Interface(fn=query_similar_images, 
                         inputs=gr.Image(type="file"), 
                         outputs=gr.Gallery(label="Imágenes similares"),
                         title=title, description=description,
                         examples=[])

if __name__ == "__main__":
    # Ejecutar Gradio en local
    interface.launch(server_name="0.0.0.0", server_port=7860)
```

En este código:
- La función `query_similar_images` es la que Gradio invoca cuando el usuario envía una imagen. Toma `input_image` (que Gradio proporciona ya sea como ruta temporal o como objeto PIL/numpy dependiendo de config). Convertimos eso a bytes (`img_data`).
- Hacemos una petición `requests.post` a la API (suponemos que la API está accesible en localhost:8000; en Docker Compose se usará probablemente el nombre del servicio).
- Parseamos el JSON de resultados. Luego, para cada resultado, hacemos otra petición GET al endpoint `/image/{id}` para obtener el contenido binario de la imagen similar.
- Creamos objetos PIL `Image` a partir del contenido y los agregamos a la lista `sim_images`.
- Retornamos `sim_images`. Gradio, al tener el output definido como `gr.Gallery`, mostrará automáticamente estas imágenes en formato de galería.

La interfaz incluye un título y descripción en español. Podemos añadir ejemplos predefinidos si quisiéramos (urls locales a algunas imágenes de ejemplo para que el usuario pruebe sin subir), pero eso es opcional.

**Integración con FastAPI**: 
- Podemos correr Gradio en un proceso separado (por ejemplo, en Docker Compose tener un servicio para la UI Gradio escuchando en el puerto 7860). 
- Alternativamente, integrar Gradio en FastAPI es posible montando la app Gradio en un endpoint de FastAPI. Gradio ofrece `mount_gradio_app`. Por ejemplo:
  ```python
  from fastapi import FastAPI
  import gradio as gr
  app = FastAPI()
  # ... definir interface de gradio ...
  app = gr.mount_gradio_app(app, interface, path="/gradio")
  ```
  De esta forma, la interfaz estaría disponible en la ruta `/gradio` del servidor FastAPI (en el mismo puerto 8000). Esto simplifica el despliegue (un solo contenedor para API+UI). En nuestro caso podríamos hacerlo para no tener contenedores separados. Por simplicidad, supongamos que usamos esta técnica para incluir Gradio en la misma aplicación FastAPI.

Usando la integración, modificaríamos el `main.py` de FastAPI así al final:

```python
# ... código anterior de FastAPI (endpoints) ...

# Integrar Gradio interface en FastAPI (sirviéndola en '/ui')
from app_gradio import interface  # asumiendo interface está creado
app = gr.mount_gradio_app(app, interface, path="/ui")
```

Con esto, cuando el sistema esté corriendo, un usuario podría abrir `http://<host>:8000/ui` y ver la interfaz gráfica. La interfaz se comunica con la propia API interna (o directamente con funciones según implementación), ofreciendo una experiencia sencilla: el usuario sube una imagen y ve una galería de imágenes similares.

## 7. Infraestructura y Despliegue (Docker Compose)

Para facilitar el despliegue reproducible del sistema, utilizaremos **Docker Compose** para orquestar los servicios:
- El servicio de la aplicación (FastAPI + modelo + Gradio).
- El servicio de base de datos de vectores (Postgres, Milvus o Manticore, según la elección).
- Configuración de volúmenes persistentes para datos (base de datos, imágenes).
- **Nota:** Se omiten integraciones con GitHub Actions u otros CI/CD, conforme a la indicación de no incluirlo.

A continuación, se describen los archivos de configuración necesarios:

**Dockerfile de la aplicación (`Dockerfile`)** – construye una imagen con Python 3.12, instala las dependencias (FastAPI, uvicorn, torch/clip, etc.), copia el código de la aplicación y establece el comando de arranque:

```dockerfile
# Usar imagen base de Python 3.12 slim (más ligera que full)
FROM python:3.12-slim

# Instalar dependencias del sistema (ej: libgl para PIL/OpenCV si se usa, git si se necesita CLIP)
RUN apt-get update && apt-get install -y \ 
    build-essential \ 
    libgl1-mesa-glx \ 
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación al contenedor
COPY . /app

# Exponer el puerto (FastAPI por defecto correrá en 8000)
EXPOSE 8000

# Comando de arranque: uvicorn para servir la app FastAPI (que incluye Gradio montado)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Archivo de requerimientos (`requirements.txt`)** – lista de paquetes Python necesarios:
```
fastapi==0.95.2
uvicorn[standard]==0.22.0
pillow==9.5.0
torch==2.0.1        # Versión de PyTorch compatible con Python 3.12
clip==0.2.0         # Suponiendo existe un paquete clip (o using openai's)
# Alternativamente, if using openai CLIP from github: could use git+https://...
# For vector DB clients:
psycopg2-binary==2.9.6    # for PostgreSQL
pymilvus==2.2.11          # Milvus client
pymysql==1.0.3            # MySQL client for Manticore
gradio==3.41.2
requests==2.31.0
```

*(Las versiones son ejemplares; habría que ajustarlas a versiones estables recientes.)*

**Docker Compose (`docker-compose.yml`)** – define los servicios, redes y volúmenes:

```yaml
version: "3.9"
services:
  app:
    build: .
    image: image-search-app:latest
    container_name: image_search_app
    environment:
      - VECTOR_DB=${VECTOR_DB}        # postgres, milvus o manticore
      - PG_HOST=${PG_HOST}
      - PG_USER=${PG_USER}
      - PG_PASSWORD=${PG_PASSWORD}
      - PG_DATABASE=${PG_DATABASE}
      - MILVUS_HOST=${MILVUS_HOST}
      - MILVUS_COLLECTION=${MILVUS_COLLECTION}
      - MANTICORE_HOST=${MANTICORE_HOST}
      - MANTICORE_INDEX=${MANTICORE_INDEX}
      - IMAGE_FOLDER=/app/images
    volumes:
      - images-data:/app/images       # volumen para almacenar imágenes indexadas
    ports:
      - "8000:8000"                   # expone FastAPI (y Gradio UI) en host
    depends_on:
      - db

  # Servicio de base de datos de vectores (se puede alternar cual se usa)
  db:
    # Ejemplo usando PostgreSQL con pgvector
    image: ramsrib/pgvector:15   # Imagen de Postgres 15 con extensión pgvector preinstalada
    container_name: pg_vector_db
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=images_db
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    # Si quisiéramos usar Milvus en su lugar, comentar lo anterior y usar:
    # image: milvusdb/milvus:v2.2.11-ee-minimal   # Milvus standalone (Enterprise minimal) 
    # container_name: milvus_vector_db
    # ports:
    #   - "19530:19530"
    # or Manticore:
    # image: manticoresearch/manticore:6.2.2
    # container_name: manticore_vector_db
    # ports:
    #   - "9306:9306"   # SQL port
    #   - "9312:9312"   # binary port, if needed
    # volumes:
    #   - manticore-data:/var/lib/manticore

volumes:
  pgdata:
  images-data:
  # manticore-data:
```

En este `docker-compose.yml`:
- Definimos dos servicios: `app` y `db`. 
- **app**:
  - Se construye desde el Dockerfile en el contexto actual (`build: .`). 
  - Las variables de entorno (`environment`) configuran qué base de datos usar y las credenciales/hosts necesarios. Aquí están referenciadas a variables de entorno del host (usando la sintaxis `${VAR}`), que podemos definir en un archivo `.env` para Docker Compose. Por ejemplo, en `.env` podríamos tener `VECTOR_DB=postgres` para este despliegue. Si quisiéramos cambiar a Milvus, pondríamos `VECTOR_DB=milvus` y asegurarnos de ajustar la configuración del servicio `db` en consecuencia.
  - Monta un volumen `images-data` en `/app/images` dentro del contenedor, para persistir las imágenes subidas/indexadas incluso si el contenedor se reinicia. Este volumen también permite que las imágenes se compartan entre reinicios o puedan ser respaldadas.
  - Expone el puerto 8000 al host, para acceder a la API y la interfaz Gradio.
  - `depends_on: db` asegura que la base de datos arranque antes que la app (aunque podríamos necesitar logic adicional para esperar a que la BD esté lista; por simplicidad, omitido).
- **db**:
  - En el ejemplo está configurado como Postgres con pgvector. Usamos una imagen de Docker Hub que viene con la extensión instalada (`ramsrib/pgvector:15`). Se definen usuario, contraseña y nombre de BD.
  - Monta `pgdata` para datos persistentes de Postgres.
  - Expone el puerto 5432 si se quiere acceder externamente (no estrictamente necesario para funcionamiento de la app, pero útil para debugging con un cliente SQL).
  - **Alternativas**: Se muestran comentados cómo sería si en su lugar usamos Milvus o Manticore:
    - Para **Milvus**: podríamos usar la imagen `milvusdb/milvus:v2.2.11-ee-minimal` (versión 2.2.11 enterprise minimal, que es *standalone* en un solo contenedor). Milvus usaría el puerto 19530 para las operaciones. Probablemente habría que exponer también un puerto web UI o etcd si se quisiera, pero para nuestro cliente pymilvus solo 19530 es necesario.
    - Para **Manticore**: imagen `manticoresearch/manticore:6.2.2` (suponiendo versión 6.2.2). Exponemos 9306 (MySQL protocol) para que `pymysql` pueda conectar. Montamos `manticore-data` en su `/var/lib/manticore` para persistir el índice.
    - **Nota**: Solo uno de estos debería estar activo a la vez. El `VECTOR_DB` env var en la app debe coincidir. En un despliegue real, podríamos maintain tres services but that wastes resources. Más bien, el usuario editaría el compose para elegir su backend preferido.
- **volumes**: Se declaran volúmenes nombrados `pgdata`, `images-data` (y `manticore-data` si se usara). Esto hace que Docker Compose los cree y los monte en los servicios.

**Despliegue**: Con estos archivos, el despliegue se reduce a:
1. Construir y levantar los servicios: `docker-compose up --build`. Esto construirá la imagen de la app, descargará la imagen de Postgres (u otra seleccionada), y lanzará ambos contenedores.
2. La primera vez, Postgres iniciará una base de datos vacía. Habrá que crear la tabla `images` con la columna vector:
   - Podemos hacerlo conectándonos al contenedor de Postgres: `docker-compose exec db psql -U postgres -d images_db -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE TABLE images (id TEXT PRIMARY KEY, embedding VECTOR(512));"` para crear la tabla con la dimensión adecuada (512 en el ejemplo para CLIP).
   - Si fuera Milvus o Manticore, la creación del índice podría gestionarse mediante su cliente o en la inicialización de la app (por ejemplo, el handler de Milvus podría verificar si la collection existe, y si no, crearla con dimension=512). Por simplicidad, no detallamos ese código, pero es recomendable.
3. Con todo funcionando, la API estará en `http://localhost:8000`. Se puede probar indexar imágenes con los scripts o manualmente vía curl/Swagger UI (FastAPI genera documentación interactiva en `/docs`). 
4. La interfaz Gradio estaría en `http://localhost:8000/ui` (según la configuración de montaje). Allí el usuario puede cargar imágenes y obtener resultados visualmente.

Este enfoque con Docker Compose facilita también escalar o modificar componentes. Por ejemplo, podríamos añadir un servicio de **NGINX** en el compose para servir como proxy inverso si quisiéramos exponer la interfaz en un puerto 80 convencional, o manejar HTTPS. También es posible escalar horizontalmente la aplicación FastAPI si se tuviese muchos usuarios concurrentes (aunque entonces la base de datos vectorial debe ser centralizada o compartida).

## 8. Lenguaje y Frameworks Utilizados

En resumen, las tecnologías empleadas para este sistema son:
- **Python 3.12**: lenguaje principal para implementar la lógica del modelo, APIs y scripts.
- **FastAPI**: framework web moderno y de alto rendimiento para construir las APIs REST de indexación y búsqueda. Provee tipado, documentación automática y facilidad de desarrollo asíncrono.
- **Gradio**: biblioteca para crear una interfaz web sencilla destinada a demostraciones de machine learning, empleada aquí para la carga de imágenes y visualización de resultados sin necesidad de programar una frontend desde cero.
- **Docker & Docker Compose**: para contenerizar la aplicación Python y los servicios de base de datos de vectores, asegurando portabilidad y fácil despliegue. Con Compose orquestamos múltiples contenedores (app, DB) y definimos volúmenes persistentes.
- **Modelos de Deep Learning**: OpenAI CLIP (arquitectura ViT-B/32 en el ejemplo) como modelo de extracción de embeddings de imágenes. PyTorch se usa para ejecutar el modelo pre-entrenado.
- **Bibliotecas auxiliares**: `requests` para llamadas HTTP en los scripts y dentro de Gradio, `Pillow` para manejo de imágenes, clientes específicos (`psycopg2`, `pymilvus`, `pymysql`) para conectar a las distintas bases vectoriales.

Con esta arquitectura y herramientas, el sistema cumple con las características solicitadas: permite indexar eficientemente ~1 millón de imágenes en vectores, almacenarlos en una base especializada, y ofrecer búsqueda de similitud rápida a través de APIs y de una interfaz web amigable, todo desplegable de forma reproducible en contenedores.

