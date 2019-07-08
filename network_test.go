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
	net := NewNetwork([]int{2, 3, 1})

	samples := map[[2]float64][]float64 {
		[2]float64{0, 0}: []float64{0},
		[2]float64{0, 1}: []float64{1},
		[2]float64{1, 0}: []float64{1},
		[2]float64{1, 1}: []float64{0},
	}

	targetMSE := 0.005
	learningRate := 0.1
	iter := 0

	for {
		iter += 1
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

		// Report squared error every 100 iterations
		if iter % 100 == 0 {
			log.Println(`iter`, iter, `MSE`, meanSquaredError)
		}

		if meanSquaredError <= targetMSE {
			break
		}
	}

	log.Println(`reached target MSE`, targetMSE, `after`, iter, `iterations`)

	// Eval each sample
	for input, target := range samples {
		output := net.forward(input[:])
		log.Println(`Input:`, input, `Target:`, target, `Output:`, output)
	}

	if iter > 800 {
		t.Error(`took more than 800 iterations to learn XOR:`, iter)
	}
}
