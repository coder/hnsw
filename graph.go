package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/TFMV/hnsw/heap"
	"golang.org/x/exp/maps"
)

type Vector = []float32

// Node is a node in the graph.
type Node[K cmp.Ordered] struct {
	Key   K
	Value Vector
}

func MakeNode[K cmp.Ordered](key K, vec Vector) Node[K] {
	return Node[K]{Key: key, Value: vec}
}

// layerNode is a node in a layer of the graph.
type layerNode[K cmp.Ordered] struct {
	Node[K]

	// neighbors is map of neighbor keys to neighbor nodes.
	// It is a map and not a slice to allow for efficient deletes, esp.
	// when M is high.
	neighbors map[K]*layerNode[K]
}

// addNeighbor adds a o neighbor to the node, replacing the neighbor
// with the worst distance if the neighbor set is full.
func (n *layerNode[K]) addNeighbor(newNode *layerNode[K], m int, dist DistanceFunc) {
	if n == nil || newNode == nil {
		return
	}

	if n.neighbors == nil {
		n.neighbors = make(map[K]*layerNode[K], m)
	}

	n.neighbors[newNode.Key] = newNode
	if len(n.neighbors) <= m {
		return
	}

	// Find the neighbor with the worst distance.
	var (
		worstDist = float32(math.Inf(-1))
		worst     *layerNode[K]
	)
	for _, neighbor := range n.neighbors {
		if neighbor == nil {
			continue
		}
		d := dist(neighbor.Value, n.Value)
		// d > worstDist may always be false if the distance function
		// returns NaN, e.g., when the embeddings are zero.
		if d > worstDist || worst == nil {
			worstDist = d
			worst = neighbor
		}
	}

	if worst != nil {
		delete(n.neighbors, worst.Key)
		// Delete backlink from the worst neighbor.
		if worst.neighbors != nil {
			delete(worst.neighbors, n.Key)
		}
		worst.replenish(m)
	}
}

type searchCandidate[K cmp.Ordered] struct {
	node *layerNode[K]
	dist float32
}

func (s searchCandidate[K]) Less(o searchCandidate[K]) bool {
	return s.dist < o.dist
}

// search returns the layer node closest to the target node
// within the same layer.
func (n *layerNode[K]) search(
	// k is the number of candidates in the result set.
	k int,
	efSearch int,
	target Vector,
	distance DistanceFunc,
) []searchCandidate[K] {
	if n == nil || distance == nil {
		return nil
	}

	// This is a basic greedy algorithm to find the entry point at the given level
	// that is closest to the target node.
	candidates := heap.Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, efSearch))
	candidates.Push(
		searchCandidate[K]{
			node: n,
			dist: distance(n.Value, target),
		},
	)
	var (
		result  = heap.Heap[searchCandidate[K]]{}
		visited = make(map[K]bool)
	)
	result.Init(make([]searchCandidate[K], 0, k))

	// Begin with the entry node in the result set.
	result.Push(candidates.Min())
	visited[n.Key] = true

	for candidates.Len() > 0 {
		var (
			current  = candidates.Pop().node
			improved = false
		)

		if current == nil || current.neighbors == nil {
			continue
		}

		// We iterate the map in a sorted, deterministic fashion for
		// tests.
		neighborKeys := maps.Keys(current.neighbors)
		slices.Sort(neighborKeys)
		for _, neighborID := range neighborKeys {
			neighbor := current.neighbors[neighborID]
			if neighbor == nil || visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.Value, target)
			improved = improved || (result.Len() > 0 && dist < result.Min().dist)
			if result.Len() < k {
				result.Push(searchCandidate[K]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist {
				result.PopLast()
				result.Push(searchCandidate[K]{node: neighbor, dist: dist})
			}

			candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
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

func (n *layerNode[K]) replenish(m int) {
	if len(n.neighbors) >= m {
		return
	}

	// Restore connectivity by adding new neighbors.
	// Use a priority queue to find the best candidates based on distance.
	candidates := heap.Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, m*2))

	// First, collect all potential candidates (neighbors of neighbors)
	visited := make(map[K]bool)
	visited[n.Key] = true // Don't add self

	// Mark existing neighbors as visited
	for k := range n.neighbors {
		visited[k] = true
	}

	// Add neighbors of neighbors as candidates
	for _, neighbor := range n.neighbors {
		if neighbor == nil || neighbor.neighbors == nil {
			continue
		}

		for k, candidate := range neighbor.neighbors {
			if visited[k] || candidate == nil {
				continue
			}
			visited[k] = true

			// Calculate distance to this node
			dist := CosineDistance(candidate.Value, n.Value)
			candidates.Push(searchCandidate[K]{
				node: candidate,
				dist: dist,
			})
		}
	}

	// Add the best candidates until we reach the desired number of neighbors
	for candidates.Len() > 0 && len(n.neighbors) < m {
		best := candidates.Pop()
		if best.node != nil {
			n.addNeighbor(best.node, m, CosineDistance)
		}
	}
}

// isolates remove the node from the graph by removing all connections
// to neighbors.
func (n *layerNode[K]) isolate(m int) {
	if n == nil || n.neighbors == nil {
		return
	}

	for _, neighbor := range n.neighbors {
		if neighbor == nil || neighbor.neighbors == nil {
			continue
		}
		delete(neighbor.neighbors, n.Key)
		neighbor.replenish(m)
	}
}

type layer[K cmp.Ordered] struct {
	// nodes is a map of nodes IDs to nodes.
	// All nodes in a higher layer are also in the lower layers, an essential
	// property of the graph.
	//
	// nodes is exported for interop with encoding/gob.
	nodes map[K]*layerNode[K]
}

// entry returns the entry node of the layer.
// It doesn't matter which node is returned, even that the
// entry node is consistent, so we just return the first node
// in the map to avoid tracking extra state.
func (l *layer[K]) entry() *layerNode[K] {
	if l == nil {
		return nil
	}
	for _, node := range l.nodes {
		return node
	}
	return nil
}

func (l *layer[K]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World graph.
// All public parameters must be set before adding nodes to the graph.
// K is cmp.Ordered instead of of comparable so that they can be sorted.
//
// Parameter Tuning Guide:
//
// M: The maximum number of connections per node.
//   - Higher values improve search accuracy but increase memory usage and build time.
//   - Lower values reduce memory usage but may decrease search accuracy.
//   - Recommended range: 8-64, with 16 being a good default for most use cases.
//   - For high-dimensional data (>1000 dimensions), higher M values (32-64) often work better.
//   - For lower-dimensional data (<100 dimensions), lower M values (8-16) are usually sufficient.
//
// Ml: The level generation factor, controlling the graph hierarchy.
//   - Determines the probability of a node being promoted to higher layers.
//   - Lower values (e.g., 0.1) create more layers with fewer nodes in higher layers.
//   - Higher values (e.g., 0.5) create fewer layers with more nodes in higher layers.
//   - Recommended range: 0.1-0.5, with 0.25 being a good default.
//   - For very large graphs (>1M nodes), lower values (0.1-0.2) often work better.
//
// EfSearch: The size of the dynamic candidate list during search.
//   - Higher values improve search accuracy but increase search time.
//   - Lower values speed up search but may decrease accuracy.
//   - Recommended range: 20-200, with 20-50 being good defaults.
//   - For applications requiring high recall, use higher values (100-200).
//   - For applications prioritizing speed, use lower values (20-50).
//
// Distance: The distance function used to compare vectors.
//   - CosineDistance is recommended for normalized embeddings (e.g., OpenAI embeddings).
//   - EuclideanDistance is recommended for non-normalized embeddings.
//   - The choice of distance function should match the properties of your data.
//
// Memory Usage:
// The memory overhead of a graph is approximately:
//   - Base layer: n * d * 4 bytes (for the vectors themselves)
//   - Graph structure: n * log(n) * sizeof(key) * M bytes
//
// Where n is the number of vectors, d is the dimensionality, and sizeof(key) is the size of the key in bytes.
type Graph[K cmp.Ordered] struct {
	// Distance is the distance function used to compare embeddings.
	Distance DistanceFunc

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// M is the maximum number of neighbors to keep for each node.
	// A good default for OpenAI embeddings is 16.
	M int

	// Ml is the level generation factor.
	// E.g., for Ml = 0.25, each layer is 1/4 the size of the previous layer.
	Ml float64

	// EfSearch is the number of nodes to consider in the search phase.
	// 20 is a reasonable default. Higher values improve search accuracy at
	// the expense of memory.
	EfSearch int

	// layers is a slice of layers in the graph.
	layers []*layer[K]
}

func defaultRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

// NewGraph returns a new graph with default parameters, roughly designed for
// storing OpenAI embeddings.
func NewGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 20,
		Rng:      defaultRand(),
	}
}

// NewGraphWithConfig returns a new graph with the specified parameters.
// It validates the configuration and returns an error if any parameter is invalid.
func NewGraphWithConfig[K cmp.Ordered](m int, ml float64, efSearch int, distance DistanceFunc) (*Graph[K], error) {
	g := &Graph[K]{
		M:        m,
		Ml:       ml,
		Distance: distance,
		EfSearch: efSearch,
		Rng:      defaultRand(),
	}

	if err := g.Validate(); err != nil {
		return nil, err
	}

	return g, nil
}

// maxLevel returns an upper-bound on the number of levels in the graph
// based on the size of the base layer.
func maxLevel(ml float64, numNodes int) (int, error) {
	if ml == 0 {
		return 0, fmt.Errorf("ml must be greater than 0")
	}

	if numNodes == 0 {
		return 1, nil
	}

	l := math.Log(float64(numNodes))
	l /= math.Log(1 / ml)

	m := int(math.Round(l)) + 1

	return m, nil
}

// randomLevel generates a random level for a new node.
func (h *Graph[K]) randomLevel() (int, error) {
	// max avoids having to accept an additional parameter for the maximum level
	// by calculating a probably good one from the size of the base layer.
	max := 1
	if len(h.layers) > 0 {
		if h.Ml == 0 {
			return 0, fmt.Errorf("(*Graph).Ml must be greater than 0")
		}
		var err error
		max, err = maxLevel(h.Ml, h.layers[0].size())
		if err != nil {
			return 0, err
		}
	}

	for level := 0; level < max; level++ {
		if h.Rng == nil {
			h.Rng = defaultRand()
		}
		r := h.Rng.Float64()
		if r > h.Ml {
			return level, nil
		}
	}

	return max, nil
}

// Dims returns the number of dimensions in the graph, or
// 0 if the graph is empty.
func (g *Graph[K]) Dims() int {
	if len(g.layers) == 0 {
		return 0
	}
	return len(g.layers[0].entry().Value)
}

func ptr[T any](v T) *T {
	return &v
}

// Add inserts nodes into the graph.
// If another node with the same ID exists, it is replaced.
func (g *Graph[K]) Add(nodes ...Node[K]) error {
	if err := g.Validate(); err != nil {
		return err
	}

	for _, node := range nodes {
		key := node.Key
		vec := node.Value

		// Check dimensions
		if len(g.layers) > 0 {
			hasDims := g.Dims()
			if hasDims != len(vec) {
				return fmt.Errorf("embedding dimension mismatch: %d != %d", hasDims, len(vec))
			}
		}

		insertLevel, err := g.randomLevel()
		if err != nil {
			return err
		}
		// Create layers that don't exist yet.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K]{})
		}

		if insertLevel < 0 {
			return fmt.Errorf("invalid level: %d", insertLevel)
		}

		var elevator *K

		preLen := g.Len()

		// Insert node at each layer, beginning with the highest.
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]
			newNode := &layerNode[K]{
				Node: Node[K]{
					Key:   key,
					Value: vec,
				},
			}

			// Insert the new node into the layer.
			if layer.entry() == nil {
				layer.nodes = map[K]*layerNode[K]{key: newNode}
				continue
			}

			// Now at the highest layer with more than one node, so we can begin
			// searching for the best way to enter the graph.
			searchPoint := layer.entry()

			// On subsequent layers, we use the elevator node to enter the graph
			// at the best point.
			if elevator != nil {
				searchPoint = layer.nodes[*elevator]
			}

			neighborhood := searchPoint.search(g.M, g.EfSearch, vec, g.Distance)
			if len(neighborhood) == 0 {
				// This should never happen because the searchPoint itself
				// should be in the result set.
				return fmt.Errorf("no nodes found in neighborhood search")
			}

			// Re-set the elevator node for the next layer.
			elevator = ptr(neighborhood[0].node.Key)

			if insertLevel >= i {
				if _, ok := layer.nodes[key]; ok {
					g.Delete(key)
				}
				// Insert the new node into the layer.
				layer.nodes[key] = newNode
				for _, node := range neighborhood {
					// Create a bi-directional edge between the new node and the best node.
					node.node.addNeighbor(newNode, g.M, g.Distance)
					newNode.addNeighbor(node.node, g.M, g.Distance)
				}
			}
		}

		// Invariant check: the node should have been added to the graph.
		if g.Len() != preLen+1 {
			return fmt.Errorf("node not added")
		}
	}

	return nil
}

// Search finds the k nearest neighbors from the target node.
func (h *Graph[K]) Search(near Vector, k int) ([]Node[K], error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be greater than 0, got %d", k)
	}

	// Check dimensions
	if len(h.layers) > 0 {
		hasDims := h.Dims()
		if hasDims != len(near) {
			return nil, fmt.Errorf("embedding dimension mismatch: %d != %d", hasDims, len(near))
		}
	}

	if len(h.layers) == 0 {
		return nil, nil
	}

	var (
		efSearch = h.EfSearch

		elevator *K
	)

	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		searchPoint := h.layers[layer].entry()
		if elevator != nil {
			searchPoint = h.layers[layer].nodes[*elevator]
		}

		// Descending hierarchies
		if layer > 0 {
			nodes := searchPoint.search(1, efSearch, near, h.Distance)
			elevator = ptr(nodes[0].node.Key)
			continue
		}

		nodes := searchPoint.search(k, efSearch, near, h.Distance)
		out := make([]Node[K], 0, len(nodes))

		for _, node := range nodes {
			out = append(out, node.node.Node)
		}

		return out, nil
	}

	return nil, fmt.Errorf("unreachable code reached")
}

// ParallelSearch finds the k nearest neighbors from the target node using parallel processing.
// It's optimized for large graphs and high-dimensional data.
// The numWorkers parameter controls the level of parallelism. If set to 0, it defaults to
// the number of CPU cores.
func (h *Graph[K]) ParallelSearch(near Vector, k int, numWorkers int) ([]Node[K], error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be greater than 0, got %d", k)
	}

	// Check dimensions
	if len(h.layers) > 0 {
		hasDims := h.Dims()
		if hasDims != len(near) {
			return nil, fmt.Errorf("embedding dimension mismatch: %d != %d", hasDims, len(near))
		}
	}

	if len(h.layers) == 0 {
		return nil, nil
	}

	// For small graphs or low dimensions, use the sequential search
	// This avoids the overhead of parallelism for small workloads
	if len(h.layers[0].nodes) < 5000 || len(near) < 512 {
		return h.Search(near, k)
	}

	// Default to number of CPU cores if numWorkers is not specified
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	// Use the same traversal logic as the sequential search
	var (
		efSearch = h.EfSearch
		elevator *K
	)

	// First phase: descend through layers to find entry point for base layer
	for layer := len(h.layers) - 1; layer > 0; layer-- {
		searchPoint := h.layers[layer].entry()
		if elevator != nil {
			searchPoint = h.layers[layer].nodes[*elevator]
		}

		nodes := searchPoint.search(1, efSearch, near, h.Distance)
		elevator = ptr(nodes[0].node.Key)
	}

	// For the base layer, use the entry point found by traversing the upper layers
	baseLayer := h.layers[0]
	searchPoint := baseLayer.entry()
	if elevator != nil {
		searchPoint = baseLayer.nodes[*elevator]
	}

	// Use a parallel version of the search algorithm
	// This is a simplified version that focuses on parallelizing the distance calculations
	candidates := heap.Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, efSearch))
	candidates.Push(
		searchCandidate[K]{
			node: searchPoint,
			dist: h.Distance(searchPoint.Value, near),
		},
	)

	var (
		result  = heap.Heap[searchCandidate[K]]{}
		visited = make(map[K]bool)
	)
	result.Init(make([]searchCandidate[K], 0, k))

	// Begin with the entry node in the result set
	result.Push(candidates.Min())
	visited[searchPoint.Key] = true

	for candidates.Len() > 0 {
		var (
			current  = candidates.Pop().node
			improved = false
		)

		// Get all neighbors to process
		neighborKeys := maps.Keys(current.neighbors)
		slices.Sort(neighborKeys)

		// Filter out already visited neighbors
		unvisitedNeighbors := make([]*layerNode[K], 0, len(neighborKeys))
		for _, neighborID := range neighborKeys {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true
			unvisitedNeighbors = append(unvisitedNeighbors, current.neighbors[neighborID])
		}

		// If we have enough neighbors, process them in parallel
		if len(unvisitedNeighbors) >= numWorkers {
			// Calculate distances in parallel
			distChan := make(chan struct {
				node *layerNode[K]
				dist float32
			}, len(unvisitedNeighbors))

			// Divide work among workers
			var wg sync.WaitGroup
			neighborsPerWorker := (len(unvisitedNeighbors) + numWorkers - 1) / numWorkers

			for i := 0; i < numWorkers && i*neighborsPerWorker < len(unvisitedNeighbors); i++ {
				start := i * neighborsPerWorker
				end := (i + 1) * neighborsPerWorker
				if end > len(unvisitedNeighbors) {
					end = len(unvisitedNeighbors)
				}

				wg.Add(1)
				go func(neighbors []*layerNode[K]) {
					defer wg.Done()
					for _, neighbor := range neighbors {
						dist := h.Distance(neighbor.Value, near)
						distChan <- struct {
							node *layerNode[K]
							dist float32
						}{node: neighbor, dist: dist}
					}
				}(unvisitedNeighbors[start:end])
			}

			// Close channel when all workers are done
			go func() {
				wg.Wait()
				close(distChan)
			}()

			// Process results as they come in
			for res := range distChan {
				dist := res.dist
				neighbor := res.node

				improved = improved || dist < result.Min().dist
				if result.Len() < k {
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				} else if dist < result.Max().dist {
					result.PopLast()
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				}

				candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
				if candidates.Len() > efSearch {
					candidates.PopLast()
				}
			}
		} else {
			// For small number of neighbors, process sequentially
			for _, neighbor := range unvisitedNeighbors {
				dist := h.Distance(neighbor.Value, near)

				improved = improved || dist < result.Min().dist
				if result.Len() < k {
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				} else if dist < result.Max().dist {
					result.PopLast()
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				}

				candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
				if candidates.Len() > efSearch {
					candidates.PopLast()
				}
			}
		}

		// Termination condition: no improvement in distance and at least
		// k candidates in the result set.
		if !improved && result.Len() >= k {
			break
		}
	}

	// Convert to output format
	candidates = result
	out := make([]Node[K], 0, candidates.Len())
	for _, candidate := range candidates.Slice() {
		out = append(out, candidate.node.Node)
	}

	return out, nil
}

// Len returns the number of nodes in the graph.
func (h *Graph[K]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

// Delete removes a node from the graph by key.
// It tries to preserve the clustering properties of the graph by
// replenishing connectivity in the affected neighborhoods.
func (h *Graph[K]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false
	}

	var deleted bool
	for _, layer := range h.layers {
		node, ok := layer.nodes[key]
		if !ok {
			continue
		}
		delete(layer.nodes, key)
		node.isolate(h.M)
		deleted = true
	}

	return deleted
}

// Lookup returns the vector with the given key.
func (h *Graph[K]) Lookup(key K) (Vector, bool) {
	if len(h.layers) == 0 {
		return nil, false
	}

	node, ok := h.layers[0].nodes[key]
	if !ok {
		return nil, false
	}
	return node.Value, ok
}

// Validate checks if the graph configuration is valid.
// It returns an error if any parameter is invalid.
func (g *Graph[K]) Validate() error {
	if g.M <= 0 {
		return fmt.Errorf("M must be greater than 0, got %d", g.M)
	}

	if g.Ml <= 0 || g.Ml >= 1 {
		return fmt.Errorf("Ml must be between 0 and 1 (exclusive), got %f", g.Ml)
	}

	if g.EfSearch <= 0 {
		return fmt.Errorf("EfSearch must be greater than 0, got %d", g.EfSearch)
	}

	if g.Distance == nil {
		return fmt.Errorf("Distance function must be set")
	}

	return nil
}
