package hnsw

import (
	"io"
)

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

func (v Vector) WriteTo(w io.Writer) (int64, error) {
	n, err := multiBinaryWrite(w, v.id, len(v.embedding), v.embedding)
	return int64(n), err
}

func (v *Vector) ReadFrom(r io.Reader) (int64, error) {
	var embLen int
	n, err := multiBinaryRead(r, &v.id, &embLen)
	if err != nil {
		return int64(n), err
	}

	v.embedding = make([]float32, embLen)
	n, err = binaryRead(r, &v.embedding)

	return int64(n), err
}

var (
	_ io.WriterTo   = (*Vector)(nil)
	_ io.ReaderFrom = (*Vector)(nil)
)
