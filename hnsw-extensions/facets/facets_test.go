package facets

import (
	"testing"

	"github.com/TFMV/hnsw"
)

func TestFacetedGraph(t *testing.T) {
	// Create a graph with proper configuration
	graph := hnsw.NewGraph[int]()

	// Create a facet store
	store := NewMemoryFacetStore[int]()

	// Create a faceted graph
	facetedGraph := NewFacetedGraph(graph, store)

	// Add a node
	node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
	facets := []Facet{
		NewBasicFacet("category", "Electronics"),
		NewBasicFacet("price", 150.0),
	}
	facetedNode := NewFacetedNode(node, facets)

	err := facetedGraph.Add(facetedNode)
	if err != nil {
		t.Fatalf("Failed to add node: %v", err)
	}

	// Add a second node to ensure the graph has enough nodes for search
	node2 := hnsw.MakeNode(2, []float32{0.2, 0.3, 0.4})
	facets2 := []Facet{
		NewBasicFacet("category", "Electronics"),
		NewBasicFacet("price", 200.0),
	}
	facetedNode2 := NewFacetedNode(node2, facets2)

	err = facetedGraph.Add(facetedNode2)
	if err != nil {
		t.Fatalf("Failed to add second node: %v", err)
	}

	// Test Search
	query := []float32{0.1, 0.2, 0.3}
	results, err := facetedGraph.Search(query, nil, 1, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	if len(results) > 0 && results[0].Node.Key != 1 {
		t.Errorf("Expected result key to be 1, got %v", results[0].Node.Key)
	}

	// Test Delete
	deleted := facetedGraph.Delete(1)
	if !deleted {
		t.Error("Expected Delete to return true")
	}

	// Search again after deletion
	results, err = facetedGraph.Search(query, nil, 1, 2)
	if err != nil {
		t.Fatalf("Search after delete failed: %v", err)
	}

	// We should still have one result (node 2)
	if len(results) != 1 {
		t.Errorf("Expected 1 result after deletion, got %d", len(results))
	}

	if len(results) > 0 && results[0].Node.Key != 2 {
		t.Errorf("Expected result key to be 2, got %v", results[0].Node.Key)
	}
}
