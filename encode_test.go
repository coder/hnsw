package hnsw

import (
	"bytes"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGraph_ExportImport(t *testing.T) {
	g := newTestGraph[Vector]()
	for i := 0; i < 128; i++ {
		g.Add(MakeVector(strconv.Itoa(i), []float32{float32(i)}))
	}

	buf := &bytes.Buffer{}
	err := g.Export(buf)
	require.NoError(t, err)

	g2 := newTestGraph[Vector]()
	err = g2.Import(buf)
	require.NoError(t, err)

	require.Equal(t, g.Len(), g2.Len())
	// TODO: make tests robust!
	// TODO: add SavedGraph type to automate
	// boilerplate.
}
