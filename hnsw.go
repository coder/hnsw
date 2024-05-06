package hnsw

import (
	"math"
	"math/rand"
	"time"

	"github.com/ammario/hnsw/heap"
)

// Embeddable describes a type that can be embedded in a HNSW graph.
type Embeddable interface {
	// ID returns a unique identifier for the object.
	ID() string
	// Embedding returns the embedding of the object.
	// float32 is used for compatibility with OpenAI embeddings.
	Embedding() []float32
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
		worstDist float32
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
	// m is the number of neighbors in the result set.
	m int,
	efSearch int,
	target Embeddable,
	distance DistanceFunc,
) []searchCandidate[T] {
	// This is a basic greedy algorithm to find the entry point at the given level
	// that is closest to the target node.
	candidates := heap.Heap[searchCandidate[T]]{}
	candidates.Push(
		searchCandidate[T]{
			node: n,
			dist: distance(n.point.Embedding(), target.Embedding()),
		},
	)
	var (
		result  = heap.Heap[searchCandidate[T]]{}
		visited = make(map[string]bool)
	)
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

			dist := distance(neighbor.point.Embedding(), target.Embedding())
			improved = improved || dist < result.Min().dist
			if result.Len() < m {
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

		// Termination condition: no improvement in distance
		if !improved {
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

var DefaultParameters = &Parameters{
	M:        6,
	Distance: CosineSimilarity,
	Rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
}

// HNSW is a Hierarchical Navigable Small World graph.
// The zero value is an empty graph with default parameters.
// Multi-threaded access must be synchronized externally.
type HNSW[T Embeddable] struct {
	*Parameters

	layers []layer[T]
}

// maxLevel returns an upper-bound on the number of levels in the graph
// based on the size of the base layer.
func maxLevel(ml float64, nodes int) int {
	l := math.Log(float64(nodes)) / math.Log((1 / ml))
	return int(math.Round(l))
}

// randomLevel generates a random level for a new node.
func (h *HNSW[T]) randomLevel() int {
	// max avoids having to accept an additional parameter for the maximum level
	// by calculating a probably good one from the size of the base layer.
	max := 1
	if len(h.layers) > 0 {
		max = maxLevel(h.params().Ml, h.layers[0].size())
	}

	r := h.params().Rng.Float64()

	level := math.Log(r) / -math.Log(h.params().Ml)

	if level > float64(max) {
		return max
	}

	return int(math.Round(level))
}

func (h *HNSW[T]) params() *Parameters {
	if h.Parameters == nil {
		return DefaultParameters
	}

	return h.Parameters
}

func (h *HNSW[T]) Add(n Embeddable) {
	insertLevel := h.randomLevel()
	// Create layers that don't exist yet.
	for insertLevel > len(h.layers) {
		h.layers = append(h.layers, layer[T]{})
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
		nodes := searchPoint.search(m, efSearch, n, h.params().Distance)
		if len(nodes) == 0 {
			// This should never happen.
			panic("no nodes found")
		}

		// Re-set the elevator node for the next layer.
		elevator = nodes[0].node.point.ID()

		if insertLevel >= i {
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
