package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"time"

	"github.com/coder/hnsw/heap"
	"golang.org/x/exp/maps"
)

type Embedding = []float32

// Embeddable describes a type that can be embedded in a HNSW graph.
type Embeddable interface {
	// ID returns a unique identifier for the object.
	ID() string
	// Embedding returns the embedding of the object.
	// float32 is used for compatibility with OpenAI embeddings.
	Embedding() Embedding
}

// layerNode is a node in a layer of the graph.
type layerNode[T Embeddable] struct {
	point Embeddable
	// neighbors is map of neighbor IDs to neighbor nodes.
	// It is a map and not a slice to allow for efficient deletes, esp.
	// when M is high.
	neighbors map[string]*layerNode[T]
}

// addNeighbor adds a o neighbor to the node, replacing the neighbor
// with the worst distance if the neighbor set is full.
func (n *layerNode[T]) addNeighbor(newNode *layerNode[T], m int, dist DistanceFunc) {
	if n.neighbors == nil {
		n.neighbors = make(map[string]*layerNode[T], m)
	}

	n.neighbors[newNode.point.ID()] = newNode
	if len(n.neighbors) <= m {
		return
	}

	// Find the neighbor with the worst distance.
	var (
		worstDist = float32(math.Inf(-1))
		worst     *layerNode[T]
	)
	for _, neighbor := range n.neighbors {
		d := dist(neighbor.point.Embedding(), n.point.Embedding())
		if d > worstDist {
			worstDist = d
			worst = neighbor
		}
	}

	delete(n.neighbors, worst.point.ID())
	// Delete backlink from the worst neighbor.
	delete(worst.neighbors, n.point.ID())
	worst.replenish(m)
}

type searchCandidate[T Embeddable] struct {
	node *layerNode[T]
	dist float32
}

func (s searchCandidate[T]) Less(o searchCandidate[T]) bool {
	return s.dist < o.dist
}

// search returns the layer node closest to the target node
// within the same layer.
func (n *layerNode[T]) search(
	// k is the number of candidates in the result set.
	k int,
	efSearch int,
	target Embedding,
	distance DistanceFunc,
) []searchCandidate[T] {
	// This is a basic greedy algorithm to find the entry point at the given level
	// that is closest to the target node.
	candidates := heap.Heap[searchCandidate[T]]{}
	candidates.Init(make([]searchCandidate[T], 0, efSearch))
	candidates.Push(
		searchCandidate[T]{
			node: n,
			dist: distance(n.point.Embedding(), target),
		},
	)
	var (
		result  = heap.Heap[searchCandidate[T]]{}
		visited = make(map[string]bool)
	)
	result.Init(make([]searchCandidate[T], 0, k))

	// Begin with the entry node in the result set.
	result.Push(candidates.Min())
	visited[n.point.ID()] = true

	for candidates.Len() > 0 {
		var (
			current  = candidates.Pop().node
			improved = false
		)

		// We iterate the map in a sorted, deterministic fashion for
		// tests.
		neighborIDs := maps.Keys(current.neighbors)
		slices.Sort(neighborIDs)
		for _, neighborID := range neighborIDs {
			neighbor := current.neighbors[neighborID]
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.point.Embedding(), target)
			improved = improved || dist < result.Min().dist
			if result.Len() < k {
				result.Push(searchCandidate[T]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist {
				result.PopLast()
				result.Push(searchCandidate[T]{node: neighbor, dist: dist})
			}

			candidates.Push(searchCandidate[T]{node: neighbor, dist: dist})
			// Always store candidates if we haven't reached the limit.
			if candidates.Len() > efSearch {
				candidates.PopLast()
			}
		}

		// Termination condition: no improvement in distance and at least
		// kMin candidates in the result set.
		if !improved && result.Len() >= k {
			break
		}
	}

	return result.Slice()
}

func (n *layerNode[T]) replenish(m int) {
	if len(n.neighbors) >= m {
		return
	}

	// Restore connectivity by adding new neighbors.
	// This is a naive implementation that could be improved by
	// using a priority queue to find the best candidates.
	for _, neighbor := range n.neighbors {
		for id, candidate := range neighbor.neighbors {
			if _, ok := n.neighbors[id]; ok {
				// do not add duplicates
				continue
			}
			if candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, CosineDistance)
			if len(n.neighbors) >= m {
				return
			}
		}
	}
}

// isolates remove the node from the graph by removing all connections
// to neighbors.
func (n *layerNode[T]) isolate(m int) {
	for _, neighbor := range n.neighbors {
		delete(neighbor.neighbors, n.point.ID())
		neighbor.replenish(m)
	}
}

type layer[T Embeddable] struct {
	// nodes is a map of node IDs to nodes.
	// All nodes in a higher layer are also in the lower layers, an essential
	// property of the graph.
	nodes map[string]*layerNode[T]
}

// entry returns the entry node of the layer.
// It doesn't matter which node is returned, even that the
// entry node is consistent, so we just return the first node
// in the map to avoid tracking extra state.
func (l *layer[T]) entry() *layerNode[T] {
	if l == nil {
		return nil
	}
	for _, node := range l.nodes {
		return node
	}
	return nil
}

func (l *layer[T]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World graph.
// All public parameters must be set before adding nodes to the graph.
type Graph[T Embeddable] struct {
	// M is the maximum number of neighbors to keep for each node.
	M int
	// Distance is the distance function used to compare embeddings.
	Distance DistanceFunc

	// Ml is the level generation factor. E.g. 1 / log(Ml) is the probability
	// of adding a node to a level.
	Ml float64

	// EfSearch is the number of nodes to consider in the search phase.
	EfSearch int

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// layers is a slice of layers in the graph.
	layers []*layer[T]
}

// NewGraph returns a new graph with default parameters, roughly designed for
// storing OpenAI embeddings.
func NewGraph[T Embeddable]() *Graph[T] {
	return &Graph[T]{
		M:        8,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 20,
		Rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// maxLevel returns an upper-bound on the number of levels in the graph
// based on the size of the base layer.
func maxLevel(ml float64, numNodes int) int {
	if ml == 0 {
		panic("ml must be greater than 0")
	}

	if numNodes == 0 {
		return 1
	}

	l := math.Log(float64(numNodes))
	l /= math.Log(1 / ml)

	m := int(math.Round(l)) + 1

	return m
}

// randomLevel generates a random level for a new node.
func (h *Graph[T]) randomLevel() int {
	// max avoids having to accept an additional parameter for the maximum level
	// by calculating a probably good one from the size of the base layer.
	max := 1
	if len(h.layers) > 0 {
		max = maxLevel(h.Ml, h.layers[0].size())
	}

	for level := 0; level < max; level++ {
		r := h.Rng.Float64()
		if r > h.Ml {
			return level
		}
	}

	return max
}

func (g *Graph[T]) assertDims(n Embedding) {
	if len(g.layers) == 0 {
		return
	}
	hasDims := len(g.layers[0].entry().point.Embedding())
	if hasDims != len(n) {
		panic(fmt.Sprint("embedding dimension mismatch: ", hasDims, " != ", len(n)))
	}
}

// Add inserts nodes into the graph.
// If another node with the same ID exists, it is replaced.
func (g *Graph[T]) Add(nodes ...T) {
	for _, n := range nodes {
		g.assertDims(n.Embedding())
		insertLevel := g.randomLevel()
		// Create layers that don't exist yet.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[T]{})
		}

		if insertLevel < 0 {
			panic("invalid level")
		}

		var elevator string

		preLen := g.Len()

		// Insert node at each layer, beginning with the highest.
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]
			newNode := &layerNode[T]{
				point: n,
			}

			// Insert the new node into the layer.
			if layer.entry() == nil {
				layer.nodes = map[string]*layerNode[T]{n.ID(): newNode}
				continue
			}

			// Now at the highest layer with more than one node, so we can begin
			// searching for the best way to enter the graph.
			searchPoint := layer.entry()

			// On subsequent layers, we use the elevator node to enter the graph
			// at the best point.
			if elevator != "" {
				searchPoint = layer.nodes[elevator]
			}

			neighborhood := searchPoint.search(g.M, g.EfSearch, n.Embedding(), g.Distance)
			if len(neighborhood) == 0 {
				// This should never happen because the searchPoint itself
				// should be in the result set.
				panic("no nodes found")
			}

			// Re-set the elevator node for the next layer.
			elevator = neighborhood[0].node.point.ID()

			if insertLevel >= i {
				if _, ok := layer.nodes[n.ID()]; ok {
					g.Delete(n.ID())
				}
				// Insert the new node into the layer.
				layer.nodes[n.ID()] = newNode
				for _, node := range neighborhood {
					// Create a bi-directional edge between the new node and the best node.
					node.node.addNeighbor(newNode, g.M, g.Distance)
					newNode.addNeighbor(node.node, g.M, g.Distance)
				}
			}
		}

		// Invariant check: the node should have been added to the graph.
		if g.Len() != preLen+1 {
			panic("node not added")
		}
	}
}

// Search finds the k nearest neighbors to the target node.
func (h *Graph[T]) Search(near Embedding, k int) []T {
	h.assertDims(near)
	if len(h.layers) == 0 {
		return nil
	}

	var (
		efSearch = h.EfSearch

		elevator string
	)

	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		searchPoint := h.layers[layer].entry()
		if elevator != "" {
			searchPoint = h.layers[layer].nodes[elevator]
		}

		// Descending hierarchies
		if layer > 0 {
			nodes := searchPoint.search(1, efSearch, near, h.Distance)
			elevator = nodes[0].node.point.ID()
			continue
		}

		nodes := searchPoint.search(k, efSearch, near, h.Distance)
		out := make([]T, 0, len(nodes))

		for _, node := range nodes {
			out = append(out, node.node.point.(T))
		}

		return out
	}

	panic("unreachable")
}

// Len returns the number of nodes in the graph.
func (h *Graph[T]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

// Delete removes a node from the graph by ID.
// It tries to preserve the clustering properties of the graph by
// replenishing the affected neighborhoods.
func (h *Graph[T]) Delete(id string) bool {
	if len(h.layers) == 0 {
		return false
	}

	var deleted bool
	for _, layer := range h.layers {
		node, ok := layer.nodes[id]
		if !ok {
			continue
		}
		delete(layer.nodes, id)
		node.isolate(h.M)
		deleted = true
	}

	return deleted
}

// Lookup returns the node with the given ID.
func (h *Graph[T]) Lookup(id string) (T, bool) {
	var zero T
	if len(h.layers) == 0 {
		return zero, false
	}

	return h.layers[0].nodes[id].point.(T), true
}
