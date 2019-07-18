package network

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

	output1 := net.Forward(input)
	error1 := net.Error(output1, target)

	net.Backprop(input, error1, 2)

	output2 := net.Forward(input)
	error2 := net.Error(output2, target)

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
	net := NewNetwork([]int{2, 3, 1}, LeakyRELUActivation{Leak: 0.01})

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
			output := net.Forward(input)
			error := net.Error(output, target)
			net.Backprop(input, error, learningRate)

			for _, e := range error {
				meanSquaredError += math.Pow(e, 2)
			}
		}

		meanSquaredError /= float64(len(samples))

		if meanSquaredError <= targetMSE {
			break
		}

		t.Log(`iter`, iter, `mse`, meanSquaredError)
	}

	if iter > 800 {
		t.Error(`took more than 800 iterations to learn XOR:`, iter)
	}
}

func TestNetworkSnapshotAndRestoreSelf(t *testing.T) {
	net := NewNetwork([]int{2, 3, 1}, SigmoidActivation{})
	output1 := net.Forward([]float64{0, 1})

	net.Snapshot(`test-network`)

	net.Restore(`test-network`)

	output2 := net.Forward([]float64{0, 1})

	if output1[0] != output2[0] {
		t.Errorf(`output changed: expected %v, got %v`, output1, output2)
	}
}

func TestNetworkSnapshotAndRestoreNewNetwork(t *testing.T) {
	net1 := NewNetwork([]int{1, 1}, SigmoidActivation{})
	net2 := NewNetwork([]int{1, 1}, SigmoidActivation{})

	net1.Snapshot(`test-network`)
	if err := net2.Restore(`test-network`); err != nil {
		t.Errorf(`can't restore network: %s`, err)
	}

	if net1.layers[0].weights.At(0, 0) != net2.layers[0].weights.At(0, 0) {
		t.Errorf(`Weight changed. Expected %f, got %f`, net1.layers[0].weights.At(0, 0), net2.layers[0].weights.At(0, 0))
	}

	output1 := net1.Forward([]float64{1})
	output2 := net2.Forward([]float64{1})

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

func TestDocExample(t *testing.T) {
	type Sample [2][]float64  // Input and output
	samples := []Sample{
		Sample{[]float64{0, 0}, []float64{0}},
		Sample{[]float64{0, 1}, []float64{1}},
		Sample{[]float64{1, 0}, []float64{1}},
		Sample{[]float64{1, 1}, []float64{0}},
	}

	learningRate := 0.75
	targetMSE := 0.005  // Target mean squared error

	net := NewNetwork([]int{2, 3, 1}, LeakyRELUActivation{Leak: 0.01})

	for iter := 0; ; iter++ {
		meanSquaredError := float64(0)
		for _, s := range samples {
			output := net.Forward(s[0])
			error := net.Error(output, s[1])
			net.Backprop(s[0], error, learningRate)

			for _, e := range error {
				meanSquaredError += math.Pow(e, 2)
			}
		}
		meanSquaredError /= float64(len(samples))

		if meanSquaredError <= targetMSE {
			break
		}

		t.Log(`Iter`, iter, `MSE`, meanSquaredError)
	}

	// Print out network for each sample:
	for i, s := range samples {
		t.Logf(`Sample %d: %v -> %v`, i, s, net.Forward(s[0]))
	}
}
