package hnsw

import (
	"bytes"
	"math/rand"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_binaryVarint(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	i := 1337

	n, err := binaryWrite(buf, i)
	require.NoError(t, err)
	require.Equal(t, 2, n)

	// Ensure that binaryRead doesn't read past the
	// varint.
	buf.Write([]byte{0, 0, 0, 0})

	var j int
	_, err = binaryRead(buf, &j)
	require.NoError(t, err)
	require.Equal(t, 1337, j)

	require.Equal(
		t,
		[]byte{0, 0, 0, 0},
		buf.Bytes(),
	)
}

func Test_binaryWrite_string(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	s := "hello"

	n, err := binaryWrite(buf, s)
	require.NoError(t, err)
	// 5 bytes for the string, 1 byte for the length.
	require.Equal(t, 5+1, n)

	var s2 string
	_, err = binaryRead(buf, &s2)
	require.NoError(t, err)
	require.Equal(t, "hello", s2)

	require.Empty(t, buf.Bytes())
}

func verifyGraphNodes[T Embeddable](t *testing.T, g *Graph[T]) {
	for _, layer := range g.layers {
		for _, node := range layer.Nodes {
			for neighborKey, neighbor := range node.neighbors {
				_, ok := layer.Nodes[neighbor.Point.ID()]
				if !ok {
					t.Errorf(
						"node %s has neighbor %s, but neighbor does not exist",
						node.Point.ID(), neighbor.Point.ID(),
					)
				}

				if neighborKey != neighbor.Point.ID() {
					t.Errorf("node %s has neighbor %s, but neighbor key is %s", node.Point.ID(),
						neighbor.Point.ID(),
						neighborKey,
					)
				}
			}
		}
	}
}

func TestGraph_ExportImport(t *testing.T) {
	rng := rand.New(rand.NewSource(0))

	g1 := newTestGraph[Vector]()
	for i := 0; i < 128; i++ {
		g1.Add(MakeVector(strconv.Itoa(i), []float32{rng.Float32()}))
	}

	buf := &bytes.Buffer{}
	err := g1.Export(buf)
	require.NoError(t, err)

	g2 := newTestGraph[Vector]()
	err = g2.Import(buf)
	require.NoError(t, err)

	require.Equal(t, g1.Len(), g2.Len())

	a1 := Analyzer[Vector]{g1}
	a2 := Analyzer[Vector]{g2}

	require.Equal(
		t,
		a1.Topography(),
		a2.Topography(),
	)

	require.Equal(
		t,
		a1.Connectivity(),
		a2.Connectivity(),
	)

	n1 := g1.Search(
		[]float32{0.5},
		10,
	)

	n2 := g2.Search(
		[]float32{0.5},
		10,
	)

	require.Equal(t, n1, n2)

	verifyGraphNodes[Vector](t, g1)
	verifyGraphNodes[Vector](t, g2)

	// TODO: make tests robust!
	// TODO: add SavedGraph type to automate
	// boilerplate.
}
