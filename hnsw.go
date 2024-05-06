package hnsw

import "math/rand"

// Embeddable describes a type that can be embedded in a HNSW graph.
type Embeddable interface {
	// Embedding returns the embedding of the object.
	// float32 is used for compatibility with OpenAI embeddings.
	Embedding() []float32
}

type node[T Embeddable] struct {
	g *HNSW[T]

	point     Embeddable
	neighbors []*node[T]
}

type layer[T Embeddable] struct {
	nodes []T
}

type Parameters struct {
	// M is the maximum number of neighbors to keep for each node.
	M        int
	Distance DistanceFunc
	// Ml is the level generation factor. E.g. 1 / log(Ml) is the probability
	// of adding a node to a level.
	Ml float64

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand
}

var DefaultParameters = &Parameters{
	M:        6,
	Distance: CosineSimilarity,
}

// HNSW is a Hierarchical Navigable Small World graph.
// The zero value is an empty graph with default parameters.
// Multi-threaded access must be synchronized externally.
type HNSW[T Embeddable] struct {
	*Parameters

	layers []layer[T]
}

func (h *HNSW[T]) params() *Parameters {
	if h.Parameters == nil {
		return DefaultParameters
	}

	return h.Parameters
}
