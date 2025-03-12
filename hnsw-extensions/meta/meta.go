// Package meta provides extensions to the HNSW library for storing and retrieving
// JSON metadata alongside vectors.
package meta

import (
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/TFMV/hnsw"
)

// MetadataNode extends the basic HNSW Node with JSON metadata.
type MetadataNode[K cmp.Ordered] struct {
	Node     hnsw.Node[K]
	Metadata json.RawMessage
}

// NewMetadataNode creates a new MetadataNode with the given node and metadata.
func NewMetadataNode[K cmp.Ordered](node hnsw.Node[K], metadata interface{}) (MetadataNode[K], error) {
	var rawMetadata json.RawMessage
	var err error

	switch m := metadata.(type) {
	case json.RawMessage:
		// Already in the right format
		rawMetadata = m
	case []byte:
		// Validate that it's valid JSON
		if !json.Valid(m) {
			return MetadataNode[K]{}, fmt.Errorf("invalid JSON metadata")
		}
		rawMetadata = m
	case string:
		// Validate that it's valid JSON
		if !json.Valid([]byte(m)) {
			return MetadataNode[K]{}, fmt.Errorf("invalid JSON metadata string")
		}
		rawMetadata = json.RawMessage(m)
	default:
		// Convert to JSON
		rawMetadata, err = json.Marshal(metadata)
		if err != nil {
			return MetadataNode[K]{}, fmt.Errorf("failed to marshal metadata: %w", err)
		}
	}

	return MetadataNode[K]{
		Node:     node,
		Metadata: rawMetadata,
	}, nil
}

// GetMetadataAs unmarshals the metadata into the provided target.
func (n MetadataNode[K]) GetMetadataAs(target interface{}) error {
	return json.Unmarshal(n.Metadata, target)
}

// SearchResult represents a search result from the HNSW graph.
// This is a copy of the type from the HNSW package to avoid import cycles.
type SearchResult[K cmp.Ordered] struct {
	Key  K
	Dist float32
}

// MetadataSearchResult extends the basic HNSW SearchResult with metadata.
type MetadataSearchResult[K cmp.Ordered] struct {
	SearchResult[K]
	Metadata json.RawMessage
}

// GetMetadataAs unmarshals the metadata into the provided target.
func (r MetadataSearchResult[K]) GetMetadataAs(target interface{}) error {
	return json.Unmarshal(r.Metadata, target)
}

// MetadataStore is an interface for storing and retrieving metadata.
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

// MemoryMetadataStore is an in-memory implementation of MetadataStore.
type MemoryMetadataStore[K cmp.Ordered] struct {
	metadata map[K]json.RawMessage
}

// NewMemoryMetadataStore creates a new in-memory metadata store.
func NewMemoryMetadataStore[K cmp.Ordered]() *MemoryMetadataStore[K] {
	return &MemoryMetadataStore[K]{
		metadata: make(map[K]json.RawMessage),
	}
}

// Add adds metadata for a key.
func (s *MemoryMetadataStore[K]) Add(key K, metadata json.RawMessage) error {
	s.metadata[key] = metadata
	return nil
}

// Get retrieves metadata for a key.
func (s *MemoryMetadataStore[K]) Get(key K) (json.RawMessage, bool) {
	metadata, ok := s.metadata[key]
	return metadata, ok
}

// Delete removes metadata for a key.
func (s *MemoryMetadataStore[K]) Delete(key K) bool {
	_, ok := s.metadata[key]
	if ok {
		delete(s.metadata, key)
	}
	return ok
}

// BatchAdd adds metadata for multiple keys.
func (s *MemoryMetadataStore[K]) BatchAdd(keys []K, metadatas []json.RawMessage) error {
	if len(keys) != len(metadatas) {
		return fmt.Errorf("keys and metadatas must have the same length")
	}

	for i, key := range keys {
		s.metadata[key] = metadatas[i]
	}
	return nil
}

// BatchGet retrieves metadata for multiple keys.
func (s *MemoryMetadataStore[K]) BatchGet(keys []K) []json.RawMessage {
	result := make([]json.RawMessage, len(keys))
	for i, key := range keys {
		result[i] = s.metadata[key]
	}
	return result
}

// BatchDelete removes metadata for multiple keys.
func (s *MemoryMetadataStore[K]) BatchDelete(keys []K) []bool {
	result := make([]bool, len(keys))
	for i, key := range keys {
		_, ok := s.metadata[key]
		if ok {
			delete(s.metadata, key)
			result[i] = true
		}
	}
	return result
}

// MetadataError represents an error related to metadata operations.
type MetadataError struct {
	Message string
}

// Error returns the error message.
func (e MetadataError) Error() string {
	return fmt.Sprintf("metadata error: %s", e.Message)
}
