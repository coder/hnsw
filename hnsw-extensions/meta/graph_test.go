package meta

import (
	"encoding/json"
	"testing"

	"github.com/TFMV/hnsw"
)

func TestMetadataGraph(t *testing.T) {
	// Create a graph
	graph := hnsw.NewGraph[int]()

	// Create a metadata store
	store := NewMemoryMetadataStore[int]()

	// Create a metadata graph
	metadataGraph := NewMetadataGraph(graph, store)

	// Create nodes with metadata
	nodes := []MetadataNode[int]{
		createTestNode(1, []float32{1.0, 0.0, 0.0}, map[string]interface{}{
			"name":     "Node 1",
			"category": "Electronics",
			"price":    999.99,
		}, t),
		createTestNode(2, []float32{0.0, 1.0, 0.0}, map[string]interface{}{
			"name":     "Node 2",
			"category": "Clothing",
			"price":    49.99,
		}, t),
		createTestNode(3, []float32{0.0, 0.0, 1.0}, map[string]interface{}{
			"name":     "Node 3",
			"category": "Books",
			"price":    19.99,
		}, t),
	}

	// Test Add
	for _, node := range nodes {
		err := metadataGraph.Add(node)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", node.Node.Key, err)
		}
	}

	// Test Get
	for _, expectedNode := range nodes {
		retrievedNode, ok := metadataGraph.Get(expectedNode.Node.Key)
		if !ok {
			t.Fatalf("Failed to get node %d", expectedNode.Node.Key)
		}

		if retrievedNode.Node.Key != expectedNode.Node.Key {
			t.Errorf("Expected key %d, got %d", expectedNode.Node.Key, retrievedNode.Node.Key)
		}

		// Check metadata
		var expectedMetadata, retrievedMetadata map[string]interface{}
		err := json.Unmarshal(expectedNode.Metadata, &expectedMetadata)
		if err != nil {
			t.Fatalf("Failed to unmarshal expected metadata: %v", err)
		}

		err = json.Unmarshal(retrievedNode.Metadata, &retrievedMetadata)
		if err != nil {
			t.Fatalf("Failed to unmarshal retrieved metadata: %v", err)
		}

		if retrievedMetadata["name"] != expectedMetadata["name"] {
			t.Errorf("Expected name %v, got %v", expectedMetadata["name"], retrievedMetadata["name"])
		}

		if retrievedMetadata["category"] != expectedMetadata["category"] {
			t.Errorf("Expected category %v, got %v", expectedMetadata["category"], retrievedMetadata["category"])
		}
	}

	// Test Search
	query := []float32{1.0, 0.1, 0.1}
	results, err := metadataGraph.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// The closest node should be Node 1
	if results[0].Key != 1 {
		t.Errorf("Expected first result to be Node 1, got Node %d", results[0].Key)
	}

	// Check metadata in search results
	var metadata map[string]interface{}
	err = results[0].GetMetadataAs(&metadata)
	if err != nil {
		t.Fatalf("Failed to get metadata from search result: %v", err)
	}

	if metadata["name"] != "Node 1" {
		t.Errorf("Expected name to be 'Node 1', got '%v'", metadata["name"])
	}

	// Test Delete
	deleted := metadataGraph.Delete(1)
	if !deleted {
		t.Error("Expected Delete to return true")
	}

	// Verify node is deleted
	_, ok := metadataGraph.Get(1)
	if ok {
		t.Error("Expected node 1 to be deleted")
	}

	// Test BatchAdd
	newNodes := []MetadataNode[int]{
		createTestNode(4, []float32{0.5, 0.5, 0.0}, map[string]interface{}{
			"name":     "Node 4",
			"category": "Electronics",
			"price":    799.99,
		}, t),
		createTestNode(5, []float32{0.5, 0.0, 0.5}, map[string]interface{}{
			"name":     "Node 5",
			"category": "Clothing",
			"price":    59.99,
		}, t),
	}

	err = metadataGraph.BatchAdd(newNodes)
	if err != nil {
		t.Fatalf("BatchAdd failed: %v", err)
	}

	// Verify nodes were added
	for _, node := range newNodes {
		_, ok := metadataGraph.Get(node.Node.Key)
		if !ok {
			t.Errorf("Expected to find node %d after batch add", node.Node.Key)
		}
	}

	// Test BatchDelete
	keys := []int{2, 3, 4, 5}
	deleteResults := metadataGraph.BatchDelete(keys)
	if len(deleteResults) != 4 {
		t.Fatalf("Expected 4 results, got %d", len(deleteResults))
	}

	for i, key := range keys {
		if !deleteResults[i] {
			t.Errorf("Expected result for key %d to be true", key)
		}

		// Verify node is deleted
		_, ok := metadataGraph.Get(key)
		if ok {
			t.Errorf("Expected node %d to be deleted", key)
		}
	}
}

// Helper function to create a test node with metadata
func createTestNode(key int, vector []float32, metadata map[string]interface{}, t *testing.T) MetadataNode[int] {
	node := hnsw.MakeNode(key, vector)
	metadataNode, err := NewMetadataNode(node, metadata)
	if err != nil {
		t.Fatalf("Failed to create metadata node: %v", err)
	}
	return metadataNode
}
