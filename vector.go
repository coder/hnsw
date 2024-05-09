package hnsw

import (
	"bytes"
	"encoding/gob"
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

func (v *Vector) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	buf.Grow(8 + len(v.id) + 4*len(v.embedding))
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(v.id)
	if err != nil {
		return nil, err
	}

	err = enc.Encode(v.embedding)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (v *Vector) GobDecode(data []byte) error {
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	err := dec.Decode(&v.id)
	if err != nil {
		return err
	}

	return dec.Decode(&v.embedding)
}

var (
	_ gob.GobDecoder = (*Vector)(nil)
	_ gob.GobEncoder = (*Vector)(nil)
)

func init() {
	gob.Register(Vector{})
}
