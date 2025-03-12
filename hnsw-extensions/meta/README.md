# Metadata Extension for HNSW

This extension adds support for storing and retrieving JSON metadata alongside vectors in HNSW graphs.

## Features

- Store arbitrary JSON metadata with each vector
- Retrieve metadata along with search results
- Support for all HNSW search operations (regular search, search with negative examples, batch search)
- Memory-efficient storage using `json.RawMessage`
- Type-safe implementation using Go generics

## Usage

### Basic Usage

```go
// Create a graph and metadata store
graph := hnsw.NewGraph[int]()
store := meta.NewMemoryMetadataStore[int]()
metadataGraph := meta.NewMetadataGraph(graph, store)

// Create a node with metadata
node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
metadata := map[string]interface{}{
    "name":     "Product 1",
    "category": "Electronics",
    "price":    999.99,
    "tags":     []string{"smartphone", "5G", "camera"},
}

// Add the node with metadata
metadataNode, err := meta.NewMetadataNode(node, metadata)
if err != nil {
    log.Fatalf("Failed to create metadata node: %v", err)
}

err = metadataGraph.Add(metadataNode)
if err != nil {
    log.Fatalf("Failed to add node: %v", err)
}

// Search with metadata
query := []float32{0.1, 0.2, 0.3}
results, err := metadataGraph.Search(query, 10)
if err != nil {
    log.Fatalf("Search failed: %v", err)
}

// Access metadata in search results
for i, result := range results {
    var metadata map[string]interface{}
    err := result.GetMetadataAs(&metadata)
    if err != nil {
        log.Printf("Failed to get metadata for result %d: %v", i, err)
        continue
    }
    
    fmt.Printf("Result %d: %s - $%.2f (%s)\n", 
        i+1, 
        metadata["name"], 
        metadata["price"], 
        metadata["category"],
    )
}
```

### Search with Negative Examples

```go
// Search with a negative example
query := []float32{0.1, 0.2, 0.3}
negative := []float32{0.9, 0.8, 0.7}
results, err := metadataGraph.SearchWithNegative(query, negative, 10, 0.7)
if err != nil {
    log.Fatalf("Search with negative failed: %v", err)
}

// Process results as before
```

### Batch Operations

```go
// Batch add nodes with metadata
nodes := []meta.MetadataNode[int]{
    createNode(1, vector1, metadata1),
    createNode(2, vector2, metadata2),
    createNode(3, vector3, metadata3),
}

err := metadataGraph.BatchAdd(nodes)
if err != nil {
    log.Fatalf("Batch add failed: %v", err)
}

// Batch search
queries := []hnsw.Vector{query1, query2, query3}
batchResults, err := metadataGraph.BatchSearch(queries, 10)
if err != nil {
    log.Fatalf("Batch search failed: %v", err)
}

// Process batch results
for i, results := range batchResults {
    fmt.Printf("Results for query %d:\n", i+1)
    for j, result := range results {
        // Process each result
    }
}
```

## Custom Metadata Types

You can use any JSON-serializable type for metadata:

```go
// Define a custom metadata type
type ProductMetadata struct {
    Name        string   `json:"name"`
    Category    string   `json:"category"`
    Price       float64  `json:"price"`
    Tags        []string `json:"tags"`
    InStock     bool     `json:"inStock"`
    ReleaseDate string   `json:"releaseDate"`
}

// Create metadata
metadata := ProductMetadata{
    Name:        "Smartphone X",
    Category:    "Electronics",
    Price:       999.99,
    Tags:        []string{"smartphone", "5G", "camera"},
    InStock:     true,
    ReleaseDate: "2023-01-15",
}

// Create and add node
node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
metadataNode, _ := meta.NewMetadataNode(node, metadata)
metadataGraph.Add(metadataNode)

// Later, retrieve and use the typed metadata
var productMetadata ProductMetadata
result.GetMetadataAs(&productMetadata)
fmt.Printf("Product: %s, Price: $%.2f\n", productMetadata.Name, productMetadata.Price)
```

## Custom Metadata Stores

You can implement your own metadata store by implementing the `MetadataStore` interface:

```go
type MetadataStore[K cmp.Ordered] interface {
    // Add adds metadata for a key.
    Add(key K, metadata json.RawMessage) error

    // Get retrieves metadata for a key.
    Get(key K) (json.RawMessage, bool)

    // Delete removes metadata for a key.
    Delete(key K) bool

    // BatchAdd adds metadata for multiple keys.
    BatchAdd(keys []K, metadatas []json.RawMessage) error

    // BatchGet retrieves metadata for multiple keys.
    BatchGet(keys []K) []json.RawMessage

    // BatchDelete removes metadata for multiple keys.
    BatchDelete(keys []K) []bool
}
```

This allows you to store metadata in different backends, such as databases or file systems.
