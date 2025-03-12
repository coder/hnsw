# Faceted Search

The `facets` package extends HNSW with faceted search capabilities, allowing you to combine vector similarity search with traditional attribute-based filtering.

## Features

- **Faceted Nodes**: Extend HNSW nodes with facets (attributes)
- **Facet Filtering**: Filter search results based on facet values
- **Combined Search**: Perform vector similarity search and facet filtering in a single operation
- **Negative Examples**: Support for negative examples in faceted search
- **Facet Aggregations**: Get aggregated statistics about facet values

### Usage

```go
// Create a new HNSW graph
graph := hnsw.NewGraph[int]()

// Create a facet store
store := facets.NewMemoryFacetStore[int]()

// Create a faceted graph
facetedGraph := facets.NewFacetedGraph(graph, store)

// Create a node with facets
node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
nodeFacets := []facets.Facet{
    facets.NewBasicFacet("category", "Electronics"),
    facets.NewBasicFacet("price", 999.99),
    facets.NewBasicFacet("brand", "TechCo"),
}
facetedNode := facets.NewFacetedNode(node, nodeFacets)

// Add to faceted graph
facetedGraph.Add(facetedNode)

// Search with facet filters
priceFilter := facets.NewRangeFilter("price", 0, 1000)
categoryFilter := facets.NewEqualityFilter("category", "Electronics")

results, err := facetedGraph.Search(
    queryVector,
    []facets.FacetFilter{priceFilter, categoryFilter},
    10, // k
    2,  // expandFactor
)
```

### Facet Types

The package includes several built-in facet types:

- **BasicFacet**: Simple facet with a name and value
- **EqualityFilter**: Filter that matches facets with equal values
- **RangeFilter**: Filter that matches numeric facets within a range
- **StringContainsFilter**: Filter that matches string facets containing a substring

### Advanced Features

#### Faceted Search with Negative Examples

```go
results, err := facetedGraph.SearchWithNegative(
    queryVector,
    negativeVector,
    []facets.FacetFilter{categoryFilter},
    10,    // k
    0.7,   // negWeight
    2,     // expandFactor
)
```

#### Facet Aggregations

```go
aggregations, err := facetedGraph.GetFacetAggregations(
    queryVector,
    []facets.FacetFilter{categoryFilter},
    []string{"brand", "price"},
    100,  // k
    1,    // expandFactor
)

// Access aggregation results
for value, count := range aggregations["brand"].Values {
    fmt.Printf("Brand %v: %d products\n", value, count)
}
```

## Examples

See the `examples` directory for complete examples of how to use the extensions:

- **Product Search**: Demonstrates faceted search for e-commerce product search

## Installation

```bash
go get github.com/TFMV/hnsw-extensions
```

## License

Same as the HNSW library.
