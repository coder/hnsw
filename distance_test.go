package hnsw

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEuclideanDistance(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	require.Equal(t, float32(5.196152), EuclideanDistance(a, b))
}
