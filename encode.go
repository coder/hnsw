package hnsw

import (
	"encoding/binary"
	"fmt"
	"io"
)

// errorEncoder is a helper type to encode multiple values

var byteOrder = binary.LittleEndian

func binaryRead(r io.Reader, data interface{}) (int, error) {
	switch v := data.(type) {
	case *int:
		br, ok := r.(io.ByteReader)
		if !ok {
			return 0, fmt.Errorf("reader does not implement io.ByteReader")
		}

		i, err := binary.ReadVarint(br)
		if err != nil {
			return 0, err
		}

		*v = int(i)
		// TODO: this will usually overshoot size.
		return binary.MaxVarintLen64, nil

	case *string:
		var ln int
		_, err := binaryRead(r, &ln)
		if err != nil {
			return 0, err
		}

		s := make([]byte, ln)
		_, err = binaryRead(r, &s)
		*v = string(s)
		return len(s), err

	case io.ReaderFrom:
		n, err := v.ReadFrom(r)
		return int(n), err

	default:
		return binary.Size(data), binary.Read(r, byteOrder, data)
	}
}

func binaryWrite(w io.Writer, data any) (int, error) {
	switch v := data.(type) {
	case int:
		var buf [binary.MaxVarintLen64]byte
		n := binary.PutVarint(buf[:], int64(v))
		n, err := w.Write(buf[:n])
		return n, err
	case io.WriterTo:
		n, err := v.WriteTo(w)
		return int(n), err
	case string:
		return multiBinaryWrite(
			w,
			len(v),
			[]byte(v),
		)
	default:
		sz := binary.Size(data)
		err := binary.Write(w, byteOrder, data)
		if err != nil {
			return 0, fmt.Errorf("encoding %T: %w", data, err)
		}
		return sz, err
	}
}

func multiBinaryWrite(w io.Writer, data ...any) (int, error) {
	var written int
	for _, d := range data {
		n, err := binaryWrite(w, d)
		written += n
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

func multiBinaryRead(r io.Reader, data ...any) (int, error) {
	var read int
	for i, d := range data {
		n, err := binaryRead(r, d)
		read += n
		if err != nil {
			return read, fmt.Errorf("reading %T at index %v: %w", d, i, err)
		}
	}
	return read, nil
}

const encodingVersion = 1

// Export writes the graph to a writer.
//
// T must be encodable by encoding/binary or implement io.WriterTo.
// The underlying value type must be encodable with Gob.
func (h *Graph[T]) Export(w io.Writer) error {
	_, err := multiBinaryWrite(
		w,
		encodingVersion,
		h.M,
		h.Ml,
		h.EfSearch,
	)
	if err != nil {
		return fmt.Errorf("encode parameters: %w", err)
	}
	_, err = binaryWrite(w, len(h.layers))
	if err != nil {
		return fmt.Errorf("encode number of layers: %w", err)
	}
	for _, layer := range h.layers {
		_, err = binaryWrite(w, len(layer.Nodes))
		if err != nil {
			return fmt.Errorf("encode number of nodes: %w", err)
		}
		for _, node := range layer.Nodes {
			_, err = binaryWrite(w, node.Point)
			if err != nil {
				return fmt.Errorf("encode node point: %w", err)
			}

			if _, err = binaryWrite(w, len(node.neighbors)); err != nil {
				return fmt.Errorf("encode number of neighbors: %w", err)
			}

			for neighbor := range node.neighbors {
				_, err = binaryWrite(w, neighbor)
				if err != nil {
					return fmt.Errorf("encode neighbor %q: %w", neighbor, err)
				}
			}
		}
	}

	return nil
}

// Import reads the graph from a reader.
// T must be decodable by encoding/binary or implement io.ReaderFrom.
// The parameters do not have to be equal to the parameters
// of the exported graph. The graph will converge onto the new parameters.
func (h *Graph[T]) Import(r io.Reader) error {
	var version int
	_, err := multiBinaryRead(r, &version, &h.M, &h.Ml, &h.EfSearch)
	if err != nil {
		return err
	}

	if version != encodingVersion {
		return fmt.Errorf("incompatible encoding version: %d", version)
	}

	var nLayers int
	_, err = binaryRead(r, &nLayers)
	if err != nil {
		return err
	}

	h.layers = make([]*layer[T], nLayers)
	for i := 0; i < nLayers; i++ {
		var nNodes int
		_, err = binaryRead(r, &nNodes)
		if err != nil {
			return err
		}

		nodes := make(map[string]*layerNode[T], nNodes)
		for j := 0; j < nNodes; j++ {
			var point T
			_, err = binaryRead(r, &point)
			if err != nil {
				return fmt.Errorf("decoding node %d: %w", j, err)
			}

			var nNeighbors int
			_, err = binaryRead(r, &nNeighbors)
			if err != nil {
				return fmt.Errorf("decoding number of neighbors for node %d: %w", j, err)
			}

			neighbors := make([]string, nNeighbors)
			for k := 0; k < nNeighbors; k++ {
				var neighbor string
				_, err = binaryRead(r, &neighbor)
				if err != nil {
					return fmt.Errorf("decoding neighbor %d for node %d: %w", k, j, err)
				}
				neighbors[k] = neighbor
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
