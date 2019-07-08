package main

import (
	"log"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayerComputeGradient(t *testing.T) {
	input := mat.NewVecDense(3, []float64{-1, 0, 1})
	error := mat.NewVecDense(2, []float64{0, 0.3})

	layer := NewLayer(3, 2)
	layer.forward(input)
	layer.computeGradient(error)
}

func TestNetworkBackprop(t *testing.T) {
	net := NewNetwork([]int{2, 3, 1})

	input := []float64{0, 1}
	target := []float64{1}

	log.Println(`In:`, input, `Target:`, target)

	output1 := net.forward(input)
	error1 := net.error(output1, target)

	log.Println(`Out1`, output1, `Err1`, error1)

	net.backprop(input, error1, 2)

	output2 := net.forward(input)
	error2 := net.error(output2, target)

	log.Println(`Out2`, output2, `Err2`, error2)

	// Calculate squared errors to see if there's at least some improvement
	se1 := float64(0)
	for _, e := range error1 {
		se1 += math.Pow(e, 2)
	}
	se2 := float64(0)
	for _, e := range error2 {
		se2 += math.Pow(e, 2)
	}

	if se2 >= se1 {
		t.Errorf(`backprop failed to improve error: error1: %f, error2: %f`, se1, se2)
	}
}
