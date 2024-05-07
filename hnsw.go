package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/coder/hnsw/heap"
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

type layerNode[T Embeddable] struct {
	point     Embeddable
	neighbors []*layerNode[T]
}

// addNeighbor adds a o neighbor to the node, replacing the neighbor
// with the worst distance if the neighbor set is full.
func (n *layerNode[T]) addNeighbor(o *layerNode[T], m int, dist DistanceFunc) {
	if len(n.neighbors) < m {
		n.neighbors = append(n.neighbors, o)
		return
	}

	// Find the neighbor with the worst distance.
	var (
		worstDist = float32(math.Inf(-1))
		worstIdx  int
	)
	for i, neighbor := range n.neighbors {
		d := dist(neighbor.point.Embedding(), n.point.Embedding())
		if d > worstDist {
			worstDist = d
			worstIdx = i
		}
	}

	// Replace the worst neighbor with the new one.
	n.neighbors[worstIdx] = o
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
	kMin int,
	kMax int,
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
	result.Init(make([]searchCandidate[T], 0, kMax))
	result.Push(candidates.Min())
	for candidates.Len() > 0 {
		current := candidates.Pop().node
		if visited[current.point.ID()] {
			continue
		}

		visited[current.point.ID()] = true
		improved := false
		for _, neighbor := range current.neighbors {
			if visited[neighbor.point.ID()] {
				continue
			}

			dist := distance(neighbor.point.Embedding(), target)
			improved = improved || dist < result.Min().dist
			if result.Len() < kMax {
				result.Push(searchCandidate[T]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist {
				result.PopLast()
				result.Push(searchCandidate[T]{node: neighbor, dist: dist})
			}

			// Always store candidates if we haven't reached the limit.
			if candidates.Len() < efSearch {
				candidates.Push(searchCandidate[T]{node: neighbor, dist: dist})
			} else if dist < candidates.Max().dist {
				// Replace the worst candidate with the new neighbor.
				candidates.PopLast()
				candidates.Push(searchCandidate[T]{node: neighbor, dist: dist})
			}
		}

		// Termination condition: no improvement in distance and at least
		// kMin candidates in the result set.
		if !improved && result.Len() >= kMin {
			break
		}
	}

	return result.Slice()
}

type layer[T Embeddable] struct {
	// nodes is a map of node IDs to nodes.
	// all nodes in a higher layer are also in the lower layers, an essential
	// property of the graph.
	nodes map[string]*layerNode[T]
	entry *layerNode[T]
}

func (l *layer[T]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

type Parameters struct {
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
}

var DefaultParameters = Parameters{
	M:        6,
	Ml:       0.5,
	Distance: CosineSimilarity,
	EfSearch: 20,
	Rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
}

// HNSW is a Hierarchical Navigable Small World graph.
// The zero value is an empty graph with default parameters.
// Multi-threaded access must be synchronized externally.
type HNSW[T Embeddable] struct {
	*Parameters

	layers []*layer[T]

	dims int
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
func (h *HNSW[T]) randomLevel() int {
	// max avoids having to accept an additional parameter for the maximum level
	// by calculating a probably good one from the size of the base layer.
	max := 1
	if len(h.layers) > 0 {
		max = maxLevel(h.params().Ml, h.layers[0].size())
	}

	for level := 0; level < max; level++ {
		r := h.params().Rng.Float64()
		if r > h.params().Ml {
			return level
		}
	}

	return max
}

func (h *HNSW[T]) params() Parameters {
	if h.Parameters == nil {
		return DefaultParameters
	}
	return *h.Parameters
}

func (h *HNSW[T]) Add(n T) {
	if h.dims == 0 {
		h.dims = len(n.Embedding())
	} else if h.dims != len(n.Embedding()) {
		panic("embedding dimension mismatch")
	}
	insertLevel := h.randomLevel()
	// Create layers that don't exist yet.
	for insertLevel > len(h.layers) {
		h.layers = append(h.layers, &layer[T]{})
	}

	var elevator string

	// Insert node at each layer, beginning with the highest.
	for i := len(h.layers) - 1; i >= 0; i-- {
		layer := h.layers[i]
		newNode := &layerNode[T]{
			point: n,
		}

		// Insert the new node into the layer.
		if layer.entry == nil {
			layer.entry = newNode
			layer.nodes = map[string]*layerNode[T]{n.ID(): newNode}
			continue
		}

		var (
			m        = h.params().M
			efSearch = h.params().EfSearch
		)

		// Now at the highest layer with more than one node, so we can begin
		// searching for the best way to enter the graph.
		searchPoint := layer.entry

		// On subsequent layers, we use the elevator node to enter the graph
		// at the best point.
		if elevator != "" {
			searchPoint = layer.nodes[elevator]
		}

		nodes := searchPoint.search(m, m, efSearch, n.Embedding(), h.params().Distance)
		if len(nodes) == 0 {
			// This should never happen because the searchPoint itself
			// should be in the result set.
			panic("no nodes found")
		}

		// Re-set the elevator node for the next layer.
		elevator = nodes[0].node.point.ID()

		if insertLevel >= i {
			if _, ok := layer.nodes[n.ID()]; ok {
				panic("must implement deleting nodes that exist")
			}
			// Insert the new node into the layer.
			layer.nodes[n.ID()] = newNode
			for _, node := range nodes {
				// Create a bi-directional edge between the new node and the best node.
				node.node.addNeighbor(newNode, m, h.params().Distance)
				newNode.addNeighbor(node.node, m, h.params().Distance)
			}
		}
	}
}

func (h *HNSW[T]) Search(near Embedding, k int) []T {
	if len(near) != h.dims {
		panic(fmt.Sprint("embedding dimension mismatch: ", len(near), " != ", h.dims))
	}
	if len(h.layers) == 0 {
		return nil
	}

	var (
		efSearch = h.params().EfSearch

		elevator string
	)

	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		searchPoint := h.layers[layer].entry
		if elevator != "" {
			searchPoint = h.layers[layer].nodes[elevator]
		}

		// Descending hierarchies
		if layer > 0 {
			nodes := searchPoint.search(1, 1, efSearch, near, h.params().Distance)
			elevator = nodes[0].node.point.ID()
			continue
		}

		// At the base layer, we can return the results.
		nodes := searchPoint.search(k, k, efSearch, near, h.params().Distance)
		out := make([]T, 0, len(nodes))
		for _, node := range nodes {
			out = append(out, node.node.point.(T))
		}

		return out
	}

	panic("unreachable")
}
