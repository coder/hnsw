package hnsw

import (
	"encoding/gob"
	"fmt"
	"io"

	"golang.org/x/exp/maps"
)

// errorEncoder is a helper type to encode multiple values
// without repetitive error checking.
type errorEncoder struct {
	err error
	enc *gob.Encoder
}

func (e *errorEncoder) Encode(v interface{}) {
	if e.err != nil {
		return
	}
	e.err = e.enc.Encode(v)
}

const encodingVersion = 1

// Export writes the graph to a writer.
// The underlying value type must be encodable with Gob.
func (h *Graph[T]) Export(w io.Writer) error {
	enc := &errorEncoder{enc: gob.NewEncoder(w)}
	enc.Encode(encodingVersion)
	enc.Encode(h.M)
	enc.Encode(h.Ml)
	enc.Encode(h.EfSearch)
	if enc.err != nil {
		return fmt.Errorf("encode parameters: %w", enc.err)
	}
	enc.Encode(len(h.layers))
	for _, layer := range h.layers {
		enc.Encode(len(layer.Nodes))
		for _, node := range layer.Nodes {
			enc.Encode(node.Point)
			enc.Encode(maps.Keys(node.neighbors))
		}
	}
	return enc.err
}

// Import reads the graph from a reader.
// The parameters do not have to be equal to the parameters
// of the exported graph.
// The graph will eventually converge onto the new parameters.
func (h *Graph[T]) Import(r io.Reader) error {
	dec := gob.NewDecoder(r)
	var version int
	err := dec.Decode(&version)
	if err != nil {
		return err
	}
	if version != encodingVersion {
		return fmt.Errorf("incompatible encoding version: %d", version)
	}

	err = dec.Decode(&h.M)
	if err != nil {
		return err
	}

	err = dec.Decode(&h.Ml)
	if err != nil {
		return err
	}

	err = dec.Decode(&h.EfSearch)
	if err != nil {
		return err
	}

	var nLayers int
	err = dec.Decode(&nLayers)
	if err != nil {
		return err
	}

	h.layers = make([]*layer[T], nLayers)
	for i := 0; i < nLayers; i++ {
		var nNodes int
		err = dec.Decode(&nNodes)
		if err != nil {
			return err
		}

		nodes := make(map[string]*layerNode[T], nNodes)
		for j := 0; j < nNodes; j++ {
			var point T
			err = dec.Decode(&point)
			if err != nil {
				return fmt.Errorf("decoding node %d: %w", j, err)
			}

			var neighbors []string
			err = dec.Decode(&neighbors)
			if err != nil {
				return err
			}
			node := &layerNode[T]{
				Point:     point,
				neighbors: make(map[string]*layerNode[T]),
			}

			nodes[point.ID()] = node
			for _, neighbor := range neighbors {
				node.neighbors[neighbor] = nil
			}
		}
		// Fill in neighbor pointers
		for _, node := range nodes {
			for id := range node.neighbors {
				node.neighbors[id] = nodes[id]
			}
		}
		h.layers[i] = &layer[T]{Nodes: nodes}
	}

	return nil
}
