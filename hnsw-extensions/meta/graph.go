package meta

import (
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/TFMV/hnsw"
)

// MetadataGraph combines an HNSW graph with metadata storage.
type MetadataGraph[K cmp.Ordered] struct {
	Graph *hnsw.Graph[K]
	Store MetadataStore[K]
}

// NewMetadataGraph creates a new MetadataGraph with the given HNSW graph and metadata store.
func NewMetadataGraph[K cmp.Ordered](graph *hnsw.Graph[K], store MetadataStore[K]) *MetadataGraph[K] {
	return &MetadataGraph[K]{
		Graph: graph,
		Store: store,
	}
}

// Add adds a node with metadata to both the graph and the metadata store.
func (g *MetadataGraph[K]) Add(node MetadataNode[K]) error {
	// Add to HNSW graph
	err := g.Graph.Add(node.Node)
	if err != nil {
		return fmt.Errorf("failed to add to graph: %w", err)
	}

	// Add to metadata store
	err = g.Store.Add(node.Node.Key, node.Metadata)
	if err != nil {
		// If adding to the metadata store fails, try to remove from the graph to maintain consistency
		g.Graph.Delete(node.Node.Key)
		return fmt.Errorf("failed to add to metadata store: %w", err)
	}

	return nil
}

// BatchAdd adds multiple nodes with metadata in a single operation.
func (g *MetadataGraph[K]) BatchAdd(nodes []MetadataNode[K]) error {
	// Extract HNSW nodes
	hnswNodes := make([]hnsw.Node[K], len(nodes))
	keys := make([]K, len(nodes))
	metadatas := make([]json.RawMessage, len(nodes))

	for i, node := range nodes {
		hnswNodes[i] = node.Node
		keys[i] = node.Node.Key
		metadatas[i] = node.Metadata
	}

	// Add to HNSW graph
	err := g.Graph.BatchAdd(hnswNodes)
	if err != nil {
		return fmt.Errorf("failed to batch add to graph: %w", err)
	}

	// Add to metadata store
	err = g.Store.BatchAdd(keys, metadatas)
	if err != nil {
		// If adding to the metadata store fails, we should ideally roll back the graph additions,
		// but that's complex. For now, we'll just report the error.
		return fmt.Errorf("failed to batch add to metadata store: %w", err)
	}

	return nil
}

// Delete removes a node from both the graph and the metadata store.
func (g *MetadataGraph[K]) Delete(key K) bool {
	graphDeleted := g.Graph.Delete(key)
	storeDeleted := g.Store.Delete(key)

	// Return true if it was deleted from either store
	return graphDeleted || storeDeleted
}

// BatchDelete removes multiple nodes in a single operation.
func (g *MetadataGraph[K]) BatchDelete(keys []K) []bool {
	graphResults := g.Graph.BatchDelete(keys)
	storeResults := g.Store.BatchDelete(keys)

	// Combine results (true if deleted from either store)
	results := make([]bool, len(keys))
	for i := range keys {
		results[i] = graphResults[i] || storeResults[i]
	}

	return results
}

// Get retrieves a node with its metadata.
func (g *MetadataGraph[K]) Get(key K) (MetadataNode[K], bool) {
	// Get vector from graph
	vector, ok := g.Graph.Lookup(key)
	if !ok {
		return MetadataNode[K]{}, false
	}

	// Get metadata from store
	metadata, ok := g.Store.Get(key)
	if !ok {
		// Node exists in graph but not in metadata store
		return MetadataNode[K]{
			Node: hnsw.Node[K]{
				Key:   key,
				Value: vector,
			},
		}, true
	}

	// Return node with metadata
	return MetadataNode[K]{
		Node: hnsw.Node[K]{
			Key:   key,
			Value: vector,
		},
		Metadata: metadata,
	}, true
}

// Search performs a search and attaches metadata to results.
func (g *MetadataGraph[K]) Search(query hnsw.Vector, k int) ([]MetadataSearchResult[K], error) {
	// Search in the graph
	results, err := g.Graph.Search(query, k)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Convert hnsw.Node to SearchResult
	searchResults := make([]SearchResult[K], len(results))
	for i, result := range results {
		searchResults[i] = SearchResult[K]{
			Key:  result.Key,
			Dist: 0, // Node doesn't have Dist field, we'll use 0 as a placeholder
		}
	}

	return g.attachMetadataToResults(searchResults)
}

// SearchWithNegative performs a search with a negative example and attaches metadata to results.
func (g *MetadataGraph[K]) SearchWithNegative(query, negative hnsw.Vector, k int, negWeight float32) ([]MetadataSearchResult[K], error) {
	// Search in the graph with negative example
	results, err := g.Graph.SearchWithNegative(query, negative, k, negWeight)
	if err != nil {
		return nil, fmt.Errorf("search with negative failed: %w", err)
	}

	// Convert hnsw.Node to SearchResult
	searchResults := make([]SearchResult[K], len(results))
	for i, result := range results {
		searchResults[i] = SearchResult[K]{
			Key:  result.Key,
			Dist: 0, // Node doesn't have Dist field, we'll use 0 as a placeholder
		}
	}

	return g.attachMetadataToResults(searchResults)
}

// BatchSearch performs multiple searches in a single operation and attaches metadata to results.
func (g *MetadataGraph[K]) BatchSearch(queries []hnsw.Vector, k int) ([][]MetadataSearchResult[K], error) {
	// Batch search in the graph
	batchResults, err := g.Graph.BatchSearch(queries, k)
	if err != nil {
		return nil, fmt.Errorf("batch search failed: %w", err)
	}

	// Process each batch of results
	metadataBatchResults := make([][]MetadataSearchResult[K], len(batchResults))
	for i, results := range batchResults {
		// Convert hnsw.Node to SearchResult
		searchResults := make([]SearchResult[K], len(results))
		for j, result := range results {
			searchResults[j] = SearchResult[K]{
				Key:  result.Key,
				Dist: 0, // Node doesn't have Dist field, we'll use 0 as a placeholder
			}
		}

		metadataResults, err := g.attachMetadataToResults(searchResults)
		if err != nil {
			return nil, fmt.Errorf("failed to attach metadata to batch %d: %w", i, err)
		}
		metadataBatchResults[i] = metadataResults
	}

	return metadataBatchResults, nil
}

// BatchSearchWithNegatives performs multiple searches with negative examples in a single operation and attaches metadata to results.
func (g *MetadataGraph[K]) BatchSearchWithNegatives(queries []hnsw.Vector, negatives []hnsw.Vector, k int, negWeight float32) ([][]MetadataSearchResult[K], error) {
	if len(queries) != len(negatives) {
		return nil, fmt.Errorf("queries and negatives must have the same length")
	}

	// Perform individual searches with negatives
	metadataBatchResults := make([][]MetadataSearchResult[K], len(queries))
	for i, query := range queries {
		results, err := g.SearchWithNegative(query, negatives[i], k, negWeight)
		if err != nil {
			return nil, fmt.Errorf("search with negative failed for query %d: %w", i, err)
		}
		metadataBatchResults[i] = results
	}

	return metadataBatchResults, nil
}

// attachMetadataToResults is a helper function to attach metadata to search results.
func (g *MetadataGraph[K]) attachMetadataToResults(results []SearchResult[K]) ([]MetadataSearchResult[K], error) {
	// Extract keys from results
	keys := make([]K, len(results))
	for i, result := range results {
		keys[i] = result.Key
	}

	// Get metadata for all keys
	metadatas := g.Store.BatchGet(keys)

	// Attach metadata to results
	metadataResults := make([]MetadataSearchResult[K], len(results))
	for i, result := range results {
		metadataResults[i] = MetadataSearchResult[K]{
			SearchResult: SearchResult[K]{
				Key:  result.Key,
				Dist: result.Dist,
			},
			Metadata: metadatas[i],
		}
	}

	return metadataResults, nil
}
