package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLayerComputeGradient(t *testing.T) {
	input := mat.NewVecDense(3, []float64{-1, 0, 1})
	error := mat.NewVecDense(2, []float64{0, 0.3})

	layer := NewLayer(3, 2, SigmoidActivation{})
	layer.forward(input)
	layer.computeGradient(error)
}

func TestNetworkBackprop(t *testing.T) {
	net := NewNetwork([]int{2, 3, 1}, SigmoidActivation{})

	input := []float64{0, 1}
	target := []float64{1}

	output1 := net.forward(input)
	error1 := net.error(output1, target)

	net.backprop(input, error1, 2)

	output2 := net.forward(input)
	error2 := net.error(output2, target)

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

func TestNetworkLearnXOR(t *testing.T) {
	net := NewNetwork([]int{2, 3, 1}, LeakyRELUActivation{0.01})

	samples := map[[2]float64][]float64{
		[2]float64{0, 0}: []float64{0},
		[2]float64{0, 1}: []float64{1},
		[2]float64{1, 0}: []float64{1},
		[2]float64{1, 1}: []float64{0},
	}

	targetMSE := 0.005
	learningRate := 0.1

	var iter int

	for iter = 0; iter < 1000; iter++ {
		meanSquaredError := float64(0)

		for input, target := range samples {
			input := input[:]
			output := net.forward(input)
			error := net.error(output, target)
			net.backprop(input, error, learningRate)

			for _, e := range error {
				meanSquaredError += math.Pow(e, 2)
			}
		}

		meanSquaredError /= float64(len(samples))

		if meanSquaredError <= targetMSE {
			break
		}
	}

	if iter > 800 {
		t.Error(`took more than 800 iterations to learn XOR:`, iter)
	}
}

func TestNetworkSnapshotAndRestoreSelf(t *testing.T) {
	net := NewNetwork([]int{2, 3, 1}, SigmoidActivation{})
	output1 := net.forward([]float64{0, 1})

	net.snapshot(`test-network`)

	net.restore(`test-network`)

	output2 := net.forward([]float64{0, 1})

	if output1[0] != output2[0] {
		t.Errorf(`output changed: expected %v, got %v`, output1, output2)
	}
}

func TestNetworkSnapshotAndRestoreNewNetwork(t *testing.T) {
	net1 := NewNetwork([]int{1, 1}, SigmoidActivation{})
	net2 := NewNetwork([]int{1, 1}, SigmoidActivation{})

	net1.snapshot(`test-network`)
	if err := net2.restore(`test-network`); err != nil {
		t.Errorf(`can't restore network: %s`, err)
	}

	if net1.layers[0].weights.At(0, 0) != net2.layers[0].weights.At(0, 0) {
		t.Errorf(`Weight changed. Expected %f, got %f`, net1.layers[0].weights.At(0, 0), net2.layers[0].weights.At(0, 0))
	}

	output1 := net1.forward([]float64{1})
	output2 := net2.forward([]float64{1})

	if output1[0] != output2[0] {
		t.Errorf(`Output changed: expected %v, got %v`, output1, output2)
	}
}

func TestLayerSnapshotAndRestoreNewLayer(t *testing.T) {
	input := mat.NewVecDense(1, []float64{1})

	layer1 := NewLayer(1, 1, SigmoidActivation{})
	output1 := layer1.forward(input)
	layer1.snapshot(`test-layer`)

	layer2 := NewLayer(1, 1, SigmoidActivation{})
	layer2.restore(`test-layer`)
	output2 := layer2.forward(input)

	if layer1.weights.At(0, 0) != layer2.weights.At(0, 0) {
		t.Errorf(`Weights changed: expected %f, got %f`, layer1.weights.At(0, 0), layer2.weights.At(0, 0))
	}

	if output1.AtVec(0) != output2.AtVec(0) {
		t.Errorf(`Output changed: expected %v, got %v`, output1, output2)
	}
}
