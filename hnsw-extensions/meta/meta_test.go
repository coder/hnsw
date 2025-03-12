package meta

import (
	"encoding/json"
	"testing"

	"github.com/TFMV/hnsw"
)

func TestMetadataNode(t *testing.T) {
	// Create a node
	node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})

	// Test with map metadata
	metadata := map[string]interface{}{
		"name":     "Test Node",
		"category": "Electronics",
		"price":    99.99,
		"tags":     []string{"test", "node", "metadata"},
	}

	metadataNode, err := NewMetadataNode(node, metadata)
	if err != nil {
		t.Fatalf("Failed to create metadata node: %v", err)
	}

	// Test GetMetadataAs
	var retrievedMetadata map[string]interface{}
	err = metadataNode.GetMetadataAs(&retrievedMetadata)
	if err != nil {
		t.Fatalf("Failed to get metadata: %v", err)
	}

	// Check metadata values
	if retrievedMetadata["name"] != "Test Node" {
		t.Errorf("Expected name to be 'Test Node', got '%v'", retrievedMetadata["name"])
	}

	if retrievedMetadata["category"] != "Electronics" {
		t.Errorf("Expected category to be 'Electronics', got '%v'", retrievedMetadata["category"])
	}

	if retrievedMetadata["price"] != 99.99 {
		t.Errorf("Expected price to be 99.99, got '%v'", retrievedMetadata["price"])
	}

	// Test with JSON string metadata
	jsonStr := `{"name":"JSON Node","active":true,"count":42}`
	metadataNode, err = NewMetadataNode(node, jsonStr)
	if err != nil {
		t.Fatalf("Failed to create metadata node with JSON string: %v", err)
	}

	// Test GetMetadataAs with struct
	type TestMetadata struct {
		Name   string `json:"name"`
		Active bool   `json:"active"`
		Count  int    `json:"count"`
	}

	var testMetadata TestMetadata
	err = metadataNode.GetMetadataAs(&testMetadata)
	if err != nil {
		t.Fatalf("Failed to get metadata as struct: %v", err)
	}

	if testMetadata.Name != "JSON Node" {
		t.Errorf("Expected name to be 'JSON Node', got '%s'", testMetadata.Name)
	}

	if !testMetadata.Active {
		t.Errorf("Expected active to be true")
	}

	if testMetadata.Count != 42 {
		t.Errorf("Expected count to be 42, got %d", testMetadata.Count)
	}
}

func TestMemoryMetadataStore(t *testing.T) {
	// Create a store
	store := NewMemoryMetadataStore[int]()

	// Test Add and Get
	metadata := json.RawMessage(`{"name":"Test Node"}`)
	err := store.Add(1, metadata)
	if err != nil {
		t.Fatalf("Failed to add metadata: %v", err)
	}

	retrievedMetadata, ok := store.Get(1)
	if !ok {
		t.Fatal("Expected to find metadata for key 1")
	}

	if string(retrievedMetadata) != `{"name":"Test Node"}` {
		t.Errorf("Expected metadata to be '{\"name\":\"Test Node\"}', got '%s'", string(retrievedMetadata))
	}

	// Test Get for non-existent key
	_, ok = store.Get(2)
	if ok {
		t.Error("Expected not to find metadata for key 2")
	}

	// Test Delete
	deleted := store.Delete(1)
	if !deleted {
		t.Error("Expected Delete to return true for existing key")
	}

	_, ok = store.Get(1)
	if ok {
		t.Error("Expected not to find metadata after deletion")
	}

	// Test Delete for non-existent key
	deleted = store.Delete(2)
	if deleted {
		t.Error("Expected Delete to return false for non-existent key")
	}

	// Test BatchAdd and BatchGet
	keys := []int{1, 2, 3}
	metadatas := []json.RawMessage{
		json.RawMessage(`{"name":"Node 1"}`),
		json.RawMessage(`{"name":"Node 2"}`),
		json.RawMessage(`{"name":"Node 3"}`),
	}

	err = store.BatchAdd(keys, metadatas)
	if err != nil {
		t.Fatalf("Failed to batch add metadata: %v", err)
	}

	retrievedMetadatas := store.BatchGet(keys)
	if len(retrievedMetadatas) != 3 {
		t.Fatalf("Expected 3 metadatas, got %d", len(retrievedMetadatas))
	}

	for i, metadata := range retrievedMetadatas {
		expected := metadatas[i]
		if string(metadata) != string(expected) {
			t.Errorf("Expected metadata %d to be '%s', got '%s'", i, string(expected), string(metadata))
		}
	}

	// Test BatchDelete
	results := store.BatchDelete(keys)
	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	for i, result := range results {
		if !result {
			t.Errorf("Expected result %d to be true", i)
		}
	}

	// Verify all are deleted
	for _, key := range keys {
		_, ok := store.Get(key)
		if ok {
			t.Errorf("Expected not to find metadata for key %d after batch deletion", key)
		}
	}
}
