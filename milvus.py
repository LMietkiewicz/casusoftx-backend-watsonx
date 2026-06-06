"""milvus.py - setup and configuration for Milvus vector database"""
from pymilvus import MilvusClient, DataType, Function, FunctionType
import config

# Connect to Milvus
uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
client = MilvusClient(uri=uri)

# Drop existing collection if it exists
if client.has_collection("file_embeddings"):
  print("Reanitializing collection and deleting data...")
  client.drop_collection("file_embeddings")

# Define analyzer parameters
analyzer_params = {
  "tokenizer": "standard",
  "filter": ["lowercase"]
}

# Create collection schema
schema = MilvusClient.create_schema()

schema.add_field(
  field_name="id",
  datatype=DataType.INT64,
  is_primary=True,
  auto_id=True,
  max_length = 100
)

schema.add_field(
  field_name="file_id",
  datatype=DataType.INT64,
  max_length=30000
)

schema.add_field(
  field_name="parent_id",
  datatype=DataType.INT64,
  max_length=30000
)

schema.add_field(
  field_name="hierarchy",
  datatype=DataType.VARCHAR,
  max_length=30000
)

schema.add_field(
  field_name="type",
  datatype=DataType.VARCHAR,
  max_length=30000
)

schema.add_field(
  field_name="contents",
  datatype=DataType.VARCHAR,
  enable_analyzer=True,
  analyzer_params=analyzer_params,
  max_length=30000
)

schema.add_field(
  field_name="dense_embedding",
  datatype=DataType.FLOAT_VECTOR,
  dim=1024
)

schema.add_field(
  field_name="sparse_embedding",
  datatype=DataType.SPARSE_FLOAT_VECTOR
)

# Define sparse vector scoring function and add to schema
bm25 = Function(
  name="BM25",
  function_type=FunctionType.BM25,
  input_field_names=["contents"],
  output_field_names=["sparse_embedding"]
)

schema.add_function(bm25)

# Define indexes
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
  field_name="dense_embedding", 
  index_type="FLAT", 
  metric_type="IP"
)

index_params.add_index(
    field_name="sparse_embedding",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25", 
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.2,
        "bm25_b": 0.75
    }
)

# Create collection
client.create_collection(
  collection_name="file_embeddings",
  schema=schema,
  index_params=index_params,
  consistency_level="Strong"
)
print(f"Collection 'file_embeddings' created successfully")
