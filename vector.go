package hnsw

var _ Embeddable = Vector{}

// Vector is a struct that holds an ID and an embedding
// and implements the Embeddable interface.
type Vector struct {
	id        string
	embedding []float32
}

// MakeVector creates a new Vector with the given ID and embedding.
func MakeVector(id string, embedding []float32) Vector {
	return Vector{
		id:        id,
		embedding: embedding,
	}
}

func (v Vector) ID() string {
	return v.id
}

func (v Vector) Embedding() []float32 {
	return v.embedding
}
