package network

import (
	"math"
	"testing"

	"github.com/farhaven/nn-go/activation"
	"gonum.org/v1/gonum/mat"
)

func TestLayerComputeGradient(t *testing.T) {
	input := mat.NewVecDense(3, []float64{-1, 0, 1})
	error := mat.NewVecDense(2, []float64{0, 0.3})

	layer := newLayer(3, 2, activation.Sigmoid{})
	output := layer.forward(input)
	layer.computeGradient(error)

	t.Log("output", output)
}

func TestNetworkBackprop(t *testing.T) {
	config := []LayerConf{
		{Inputs: 2},
		{Inputs: 3, Activation: activation.Sigmoid{}},
		{Inputs: 1, Activation: activation.Sigmoid{}},
	}
	net, err := New(config)
	if err != nil {
		t.Error(`can't create network`, err)
	}

	input := []float64{0, 1}
	target := []float64{1}

	output1 := net.Forward(input)
	error1 := Error(output1, target)

	net.Backprop(input, error1, 2)

	output2 := net.Forward(input)
	error2 := Error(output2, target)

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

	t.Log("squared errors:", se1, se2)
}

func TestNetworkLearnXOR(t *testing.T) {
	act := activation.Tanh{}

	config := []LayerConf{
		{Inputs: 2},
		{Inputs: 3, Activation: act},
		{Inputs: 1, Activation: act},
	}
	net, err := New(config)
	if err != nil {
		t.Error(`can't create network`, err)
	}

	samples := map[[2]float64][]float64{
		{0, 0}: {0},
		{0, 1}: {1},
		{1, 0}: {1},
		{1, 1}: {0},
	}

	targetMSE := 0.005
	learningRate := 0.5

	var iter int

	for iter = 0; iter < 1000; iter++ {
		meanSquaredError := float64(0)

		for input, target := range samples {
			input := input[:]
			output := net.Forward(input)
			error := Error(output, target)
			net.Backprop(input, error, learningRate)

			for _, e := range error {
				meanSquaredError += math.Pow(e, 2)
			}
		}

		meanSquaredError /= float64(len(samples))

		if meanSquaredError <= targetMSE {
			break
		}

		if (iter+1)%10 == 0 {
			t.Log(`iter`, iter, `mse`, meanSquaredError)
		}
	}

	if iter > 800 {
		t.Error(`took more than 800 iterations to learn XOR:`, iter)
	} else {
		t.Log("learned XOR in", iter, "iterations")
	}

	for sample, target := range samples {
		output := net.Forward(sample[:])
		t.Log("sample", sample, "target", target, "output", output)
	}
}

func TestNetworkSnapshotAndRestoreSelf(t *testing.T) {
	config := []LayerConf{
		{Inputs: 2},
		{Inputs: 3, Activation: activation.Sigmoid{}},
		{Inputs: 1, Activation: activation.Sigmoid{}},
	}
	net, err := New(config)
	if err != nil {
		t.Error(`can't create network`, err)
	}
	output1 := net.Forward([]float64{0, 1})

	err = net.Snapshot(`test-network`)
	if err != nil {
		t.Fatal("unexpected error during snapshot:", err)
	}

	err = net.Restore(`test-network`)
	if err != nil {
		t.Fatal("unexpected error during restore:", err)
	}

	output2 := net.Forward([]float64{0, 1})

	if output1[0] != output2[0] {
		t.Errorf(`output changed: expected %v, got %v`, output1, output2)
	}
}

func TestNetworkSnapshotAndRestoreNew(t *testing.T) {
	config := []LayerConf{
		{Inputs: 2},
		{Inputs: 3, Activation: activation.Sigmoid{}},
		{Inputs: 1, Activation: activation.Sigmoid{}},
	}

	net1, err := New(config)
	if err != nil {
		t.Error(`can't create first network`, err)
	}
	net2, err := New(config)
	if err != nil {
		t.Error(`can't create second network`, err)
	}

	err = net1.Snapshot(`test-network`)
	if err != nil {
		t.Fatalf("unexpected error during snapshot: %s", err)
	}

	if err := net2.Restore(`test-network`); err != nil {
		t.Fatalf(`can't restore network: %s`, err)
	}

	if net1.layers[0].weights.At(0, 0) != net2.layers[0].weights.At(0, 0) {
		t.Errorf(`Weight changed. Expected %f, got %f`, net1.layers[0].weights.At(0, 0), net2.layers[0].weights.At(0, 0))
	}

	output1 := net1.Forward([]float64{1, 0})
	output2 := net2.Forward([]float64{1, 0})

	if output1[0] != output2[0] {
		t.Errorf(`Output changed: expected %v, got %v`, output1, output2)
	}
}

func TestLayerSnapshotAndRestoreNewLayer(t *testing.T) {
	input := mat.NewVecDense(1, []float64{1})

	layer1 := newLayer(1, 1, activation.Sigmoid{})
	output1 := layer1.forward(input)

	err := layer1.snapshot(`test-layer`)
	if err != nil {
		t.Fatal("can't snapshot layer:", err)
	}

	layer2 := newLayer(1, 1, activation.Sigmoid{})

	err = layer2.restore(`test-layer`)
	if err != nil {
		t.Fatal("can't restore layer:", err)
	}

	output2 := layer2.forward(input)

	if layer1.weights.At(0, 0) != layer2.weights.At(0, 0) {
		t.Errorf(`Weights changed: expected %f, got %f`, layer1.weights.At(0, 0), layer2.weights.At(0, 0))
	}

	if output1.AtVec(0) != output2.AtVec(0) {
		t.Errorf(`Output changed: expected %v, got %v`, output1, output2)
	}
}
