// Package facets provides extensions to the HNSW library for faceted search capabilities.
package facets

import (
	"cmp"
	"fmt"
	"reflect"
	"strings"

	"github.com/TFMV/hnsw"
)

// Facet represents a single facet (attribute) that can be attached to a vector.
type Facet interface {
	// Name returns the name of the facet.
	Name() string

	// Value returns the value of the facet.
	Value() interface{}

	// Match checks if this facet matches the given query.
	Match(query interface{}) bool
}

// FacetFilter defines a filter to be applied on facets.
type FacetFilter interface {
	// Name returns the name of the facet this filter applies to.
	Name() string

	// Matches checks if a facet value matches this filter.
	Matches(value interface{}) bool
}

// FacetedNode extends the basic HNSW Node with facets.
type FacetedNode[K cmp.Ordered] struct {
	Node   hnsw.Node[K]
	Facets []Facet
}

// NewFacetedNode creates a new FacetedNode with the given node and facets.
func NewFacetedNode[K cmp.Ordered](node hnsw.Node[K], facets []Facet) FacetedNode[K] {
	return FacetedNode[K]{
		Node:   node,
		Facets: facets,
	}
}

// GetFacet returns the facet with the given name, or nil if not found.
func (n FacetedNode[K]) GetFacet(name string) Facet {
	for _, f := range n.Facets {
		if f.Name() == name {
			return f
		}
	}
	return nil
}

// MatchesFilter checks if this node matches the given filter.
func (n FacetedNode[K]) MatchesFilter(filter FacetFilter) bool {
	facet := n.GetFacet(filter.Name())
	if facet == nil {
		return false
	}
	return filter.Matches(facet.Value())
}

// MatchesAllFilters checks if this node matches all the given filters.
func (n FacetedNode[K]) MatchesAllFilters(filters []FacetFilter) bool {
	for _, filter := range filters {
		if !n.MatchesFilter(filter) {
			return false
		}
	}
	return true
}

// BasicFacet is a simple implementation of the Facet interface.
type BasicFacet struct {
	name  string
	value interface{}
}

// NewBasicFacet creates a new BasicFacet with the given name and value.
func NewBasicFacet(name string, value interface{}) BasicFacet {
	return BasicFacet{
		name:  name,
		value: value,
	}
}

// Name returns the name of the facet.
func (f BasicFacet) Name() string {
	return f.name
}

// Value returns the value of the facet.
func (f BasicFacet) Value() interface{} {
	return f.value
}

// Match checks if this facet matches the given query.
func (f BasicFacet) Match(query interface{}) bool {
	// Simple equality check
	return reflect.DeepEqual(f.value, query)
}

// EqualityFilter is a filter that matches facets with equal values.
type EqualityFilter struct {
	name  string
	value interface{}
}

// NewEqualityFilter creates a new EqualityFilter with the given name and value.
func NewEqualityFilter(name string, value interface{}) EqualityFilter {
	return EqualityFilter{
		name:  name,
		value: value,
	}
}

// Name returns the name of the facet this filter applies to.
func (f EqualityFilter) Name() string {
	return f.name
}

// Matches checks if a facet value equals this filter's value.
func (f EqualityFilter) Matches(value interface{}) bool {
	return reflect.DeepEqual(f.value, value)
}

// RangeFilter is a filter that matches numeric facets within a range.
type RangeFilter struct {
	name string
	min  float64
	max  float64
}

// NewRangeFilter creates a new RangeFilter with the given name, min, and max values.
func NewRangeFilter(name string, min, max float64) RangeFilter {
	return RangeFilter{
		name: name,
		min:  min,
		max:  max,
	}
}

// Name returns the name of the facet this filter applies to.
func (f RangeFilter) Name() string {
	return f.name
}

// Matches checks if a numeric facet value is within the range.
func (f RangeFilter) Matches(value interface{}) bool {
	// Convert value to float64
	var floatValue float64
	switch v := value.(type) {
	case float64:
		floatValue = v
	case float32:
		floatValue = float64(v)
	case int:
		floatValue = float64(v)
	case int64:
		floatValue = float64(v)
	case int32:
		floatValue = float64(v)
	default:
		return false // Not a numeric type
	}

	return floatValue >= f.min && floatValue <= f.max
}

// StringContainsFilter is a filter that matches string facets containing a substring.
type StringContainsFilter struct {
	name     string
	contains string
}

// NewStringContainsFilter creates a new StringContainsFilter with the given name and substring.
func NewStringContainsFilter(name string, contains string) StringContainsFilter {
	return StringContainsFilter{
		name:     name,
		contains: contains,
	}
}

// Name returns the name of the facet this filter applies to.
func (f StringContainsFilter) Name() string {
	return f.name
}

// Matches checks if a string facet value contains the substring.
func (f StringContainsFilter) Matches(value interface{}) bool {
	// Convert value to string
	var strValue string
	switch v := value.(type) {
	case string:
		strValue = v
	default:
		// Try to convert to string
		strValue = fmt.Sprintf("%v", v)
	}

	return strings.Contains(strings.ToLower(strValue), strings.ToLower(f.contains))
}

// FacetStore is an interface for storing and retrieving faceted nodes.
type FacetStore[K cmp.Ordered] interface {
	// Add adds a faceted node to the store.
	Add(node FacetedNode[K]) error

	// Get retrieves a faceted node by key.
	Get(key K) (FacetedNode[K], bool)

	// Delete removes a faceted node from the store.
	Delete(key K) bool

	// Filter returns all nodes that match the given filters.
	Filter(filters []FacetFilter) []FacetedNode[K]
}

// MemoryFacetStore is an in-memory implementation of FacetStore.
type MemoryFacetStore[K cmp.Ordered] struct {
	nodes map[K]FacetedNode[K]
}

// NewMemoryFacetStore creates a new in-memory facet store.
func NewMemoryFacetStore[K cmp.Ordered]() *MemoryFacetStore[K] {
	return &MemoryFacetStore[K]{
		nodes: make(map[K]FacetedNode[K]),
	}
}

// Add adds a faceted node to the store.
func (s *MemoryFacetStore[K]) Add(node FacetedNode[K]) error {
	s.nodes[node.Node.Key] = node
	return nil
}

// Get retrieves a faceted node by key.
func (s *MemoryFacetStore[K]) Get(key K) (FacetedNode[K], bool) {
	node, ok := s.nodes[key]
	return node, ok
}

// Delete removes a faceted node from the store.
func (s *MemoryFacetStore[K]) Delete(key K) bool {
	_, ok := s.nodes[key]
	if ok {
		delete(s.nodes, key)
	}
	return ok
}

// Filter returns all nodes that match the given filters.
func (s *MemoryFacetStore[K]) Filter(filters []FacetFilter) []FacetedNode[K] {
	var result []FacetedNode[K]

	for _, node := range s.nodes {
		if node.MatchesAllFilters(filters) {
			result = append(result, node)
		}
	}

	return result
}

// FacetError represents an error related to facet operations.
type FacetError struct {
	Message string
}

// Error returns the error message.
func (e FacetError) Error() string {
	return fmt.Sprintf("facet error: %s", e.Message)
}
