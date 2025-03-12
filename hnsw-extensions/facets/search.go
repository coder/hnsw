// Package facets provides extensions to the HNSW library for faceted search capabilities.
package facets

import (
	"cmp"
	"fmt"
	"sort"

	"github.com/TFMV/hnsw"
)

// FacetedSearch performs a search with facet filtering.
// It first searches for the k*expandFactor nearest neighbors using vector similarity,
// then filters the results by the given facet filters, and finally returns the top k results.
func FacetedSearch[K cmp.Ordered](
	graph *hnsw.Graph[K],
	store FacetStore[K],
	query hnsw.Vector,
	filters []FacetFilter,
	k int,
	expandFactor int,
) ([]FacetedNode[K], error) {
	if k <= 0 {
		return nil, &FacetError{Message: "k must be greater than 0"}
	}

	if expandFactor <= 0 {
		expandFactor = 3 // Default expand factor
	}

	// Expand search to get more candidates than needed
	expandedK := k * expandFactor
	candidates, err := graph.Search(query, expandedK)
	if err != nil {
		return nil, err
	}

	// Filter candidates by facets
	var filteredNodes []FacetedNode[K]
	for _, candidate := range candidates {
		facetedNode, ok := store.Get(candidate.Key)
		if !ok {
			continue // Skip nodes not in the facet store
		}

		if facetedNode.MatchesAllFilters(filters) {
			filteredNodes = append(filteredNodes, facetedNode)
		}
	}

	// If we don't have enough results after filtering, try to get more
	if len(filteredNodes) < k && len(candidates) == expandedK {
		// We might need more candidates
		moreK := expandedK * 2
		moreCandidates, err := graph.Search(query, moreK)
		if err != nil {
			return nil, err
		}

		// Only process the new candidates
		for i := expandedK; i < len(moreCandidates); i++ {
			candidate := moreCandidates[i]
			facetedNode, ok := store.Get(candidate.Key)
			if !ok {
				continue
			}

			if facetedNode.MatchesAllFilters(filters) {
				filteredNodes = append(filteredNodes, facetedNode)
			}
		}
	}

	// Sort by distance to query (which should already be the case from the HNSW search)
	// but we'll sort again to be sure
	sort.Slice(filteredNodes, func(i, j int) bool {
		distI := graph.Distance(filteredNodes[i].Node.Value, query)
		distJ := graph.Distance(filteredNodes[j].Node.Value, query)
		return distI < distJ
	})

	// Take top k results
	if len(filteredNodes) > k {
		filteredNodes = filteredNodes[:k]
	}

	return filteredNodes, nil
}

// FacetedSearchWithNegative performs a search with facet filtering and a negative example.
// It combines vector similarity search with facet filtering and negative example avoidance.
func FacetedSearchWithNegative[K cmp.Ordered](
	graph *hnsw.Graph[K],
	store FacetStore[K],
	query hnsw.Vector,
	negative hnsw.Vector,
	filters []FacetFilter,
	k int,
	negWeight float32,
	expandFactor int,
) ([]FacetedNode[K], error) {
	if k <= 0 {
		return nil, &FacetError{Message: "k must be greater than 0"}
	}

	if negWeight < 0.0 || negWeight > 1.0 {
		return nil, &FacetError{Message: "negWeight must be between 0.0 and 1.0"}
	}

	if expandFactor <= 0 {
		expandFactor = 3 // Default expand factor
	}

	// Use the HNSW library's SearchWithNegative
	expandedK := k * expandFactor
	candidates, err := graph.SearchWithNegative(query, negative, expandedK, negWeight)
	if err != nil {
		return nil, err
	}

	// Filter candidates by facets
	var filteredNodes []FacetedNode[K]
	for _, candidate := range candidates {
		facetedNode, ok := store.Get(candidate.Key)
		if !ok {
			continue // Skip nodes not in the facet store
		}

		if facetedNode.MatchesAllFilters(filters) {
			filteredNodes = append(filteredNodes, facetedNode)
		}
	}

	// If we don't have enough results after filtering, try to get more
	if len(filteredNodes) < k && len(candidates) == expandedK {
		// We might need more candidates
		moreK := expandedK * 2
		moreCandidates, err := graph.SearchWithNegative(query, negative, moreK, negWeight)
		if err != nil {
			return nil, err
		}

		// Only process the new candidates
		for i := expandedK; i < len(moreCandidates); i++ {
			candidate := moreCandidates[i]
			facetedNode, ok := store.Get(candidate.Key)
			if !ok {
				continue
			}

			if facetedNode.MatchesAllFilters(filters) {
				filteredNodes = append(filteredNodes, facetedNode)
			}
		}
	}

	// Take top k results (they should already be sorted by the HNSW search)
	if len(filteredNodes) > k {
		filteredNodes = filteredNodes[:k]
	}

	return filteredNodes, nil
}

// FacetedGraph combines an HNSW graph with a facet store for faceted search.
type FacetedGraph[K cmp.Ordered] struct {
	Graph *hnsw.Graph[K]
	Store FacetStore[K]
}

// NewFacetedGraph creates a new FacetedGraph with the given HNSW graph and facet store.
func NewFacetedGraph[K cmp.Ordered](graph *hnsw.Graph[K], store FacetStore[K]) *FacetedGraph[K] {
	return &FacetedGraph[K]{
		Graph: graph,
		Store: store,
	}
}

// Add adds a node with facets to both the graph and the facet store.
func (fg *FacetedGraph[K]) Add(node FacetedNode[K]) error {
	// Add to HNSW graph
	err := fg.Graph.Add(node.Node)
	if err != nil {
		return fmt.Errorf("failed to add to graph: %w", err)
	}

	// Add to facet store
	err = fg.Store.Add(node)
	if err != nil {
		// If adding to the facet store fails, try to remove from the graph to maintain consistency
		fg.Graph.Delete(node.Node.Key)
		return fmt.Errorf("failed to add to facet store: %w", err)
	}

	return nil
}

// Delete removes a node from both the graph and the facet store.
func (fg *FacetedGraph[K]) Delete(key K) bool {
	graphDeleted := fg.Graph.Delete(key)
	storeDeleted := fg.Store.Delete(key)

	// Return true if it was deleted from either store
	return graphDeleted || storeDeleted
}

// Search performs a faceted search.
func (fg *FacetedGraph[K]) Search(
	query hnsw.Vector,
	filters []FacetFilter,
	k int,
	expandFactor int,
) ([]FacetedNode[K], error) {
	return FacetedSearch(fg.Graph, fg.Store, query, filters, k, expandFactor)
}

// SearchWithNegative performs a faceted search with a negative example.
func (fg *FacetedGraph[K]) SearchWithNegative(
	query hnsw.Vector,
	negative hnsw.Vector,
	filters []FacetFilter,
	k int,
	negWeight float32,
	expandFactor int,
) ([]FacetedNode[K], error) {
	return FacetedSearchWithNegative(fg.Graph, fg.Store, query, negative, filters, k, negWeight, expandFactor)
}

// BatchAdd adds multiple nodes with facets in a single operation.
func (fg *FacetedGraph[K]) BatchAdd(nodes []FacetedNode[K]) error {
	// Extract HNSW nodes
	hnswNodes := make([]hnsw.Node[K], len(nodes))
	for i, node := range nodes {
		hnswNodes[i] = node.Node
	}

	// Add to HNSW graph
	err := fg.Graph.BatchAdd(hnswNodes)
	if err != nil {
		return fmt.Errorf("failed to batch add to graph: %w", err)
	}

	// Add to facet store
	for _, node := range nodes {
		err := fg.Store.Add(node)
		if err != nil {
			// If adding to the facet store fails, we should ideally roll back the graph additions,
			// but that's complex. For now, we'll just report the error.
			return fmt.Errorf("failed to add node %v to facet store: %w", node.Node.Key, err)
		}
	}

	return nil
}

// BatchDelete removes multiple nodes in a single operation.
func (fg *FacetedGraph[K]) BatchDelete(keys []K) []bool {
	// Delete from HNSW graph
	graphResults := fg.Graph.BatchDelete(keys)

	// Delete from facet store
	storeResults := make([]bool, len(keys))
	for i, key := range keys {
		storeResults[i] = fg.Store.Delete(key)
	}

	// Combine results (true if deleted from either store)
	results := make([]bool, len(keys))
	for i := range keys {
		results[i] = graphResults[i] || storeResults[i]
	}

	return results
}

// FacetAggregation represents an aggregation of facet values.
type FacetAggregation struct {
	Name   string
	Values map[interface{}]int
}

// GetFacetAggregations returns aggregations of facet values for the given facet names.
func (fg *FacetedGraph[K]) GetFacetAggregations(
	query hnsw.Vector,
	filters []FacetFilter,
	facetNames []string,
	k int,
	expandFactor int,
) (map[string]FacetAggregation, error) {
	// Perform a search to get candidates
	expandedK := k * expandFactor
	candidates, err := fg.Graph.Search(query, expandedK)
	if err != nil {
		return nil, err
	}

	// Initialize aggregations
	aggregations := make(map[string]FacetAggregation)
	for _, name := range facetNames {
		aggregations[name] = FacetAggregation{
			Name:   name,
			Values: make(map[interface{}]int),
		}
	}

	// Collect facet values from candidates
	for _, candidate := range candidates {
		facetedNode, ok := fg.Store.Get(candidate.Key)
		if !ok {
			continue
		}

		// Skip nodes that don't match the filters
		if !facetedNode.MatchesAllFilters(filters) {
			continue
		}

		// Aggregate facet values
		for _, name := range facetNames {
			facet := facetedNode.GetFacet(name)
			if facet != nil {
				value := facet.Value()
				aggregations[name].Values[value]++
			}
		}
	}

	return aggregations, nil
}
