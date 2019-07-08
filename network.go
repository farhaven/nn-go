package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	weights         *mat.Dense
	delta           *mat.VecDense
	output          *mat.VecDense
	scratch         *mat.Dense // Scratch buffer for weight updates
	activationPrime func(float64) float64
	activation      func(float64) float64
}

func NewLayer(inputs, outputs int) Layer {
	// Assumes tanh activation
	// Initialize layer with random weights
	weights := mat.NewDense(outputs, inputs, nil)
	weights.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64()
	}, weights)

	return Layer{
		weights:    weights,
		delta:      mat.NewVecDense(outputs, nil),
		output:     mat.NewVecDense(outputs, nil),
		scratch:    mat.NewDense(outputs, inputs, nil),
		activation: math.Tanh,
		activationPrime: func(x float64) float64 {
			return 1 - math.Pow(x, 2.0)
		},
	}
}

func (l *Layer) snapshot(path string) {
	fh, err := os.Create(path)
	if err != nil {
		log.Fatalf(`can't open snapshot file %s: %s`, path, err)
	}
	defer fh.Close()

	l.weights.MarshalBinaryTo(fh)
}

func (l *Layer) restore(path string) {
	fh, err := os.Open(path)
	if err != nil {
		log.Fatalf(`can't open snapshot file %s: %s`, path, err)
	}
	defer fh.Close()

	var weights mat.Dense
	n, err := weights.UnmarshalBinaryFrom(fh)
	if err != nil {
		log.Fatalf(`can't read snapshot from %s: %s`, path, err)
	}
	log.Println(`read`, n, `bytes of snapshot data from`, path)
	l.weights = &weights
}

func (l *Layer) computeGradient(error *mat.VecDense) *mat.VecDense {
	for idx := 0; idx < error.Len(); idx++ {
		// TODO: See if this can be unified
		e := error.AtVec(idx)
		l.delta.SetVec(idx, e*l.activationPrime(l.output.AtVec(idx)))
	}

	var res mat.Dense
	res.Mul(mat.Matrix(l.delta).T(), l.weights)

	_, c := l.weights.Dims()

	resVec := mat.NewVecDense(c, nil)
	resVec.CopyVec(res.RowView(0))

	return resVec
}

func (l *Layer) forward(inputs *mat.VecDense) *mat.VecDense {
	l.output.MulVec(l.weights, inputs) // TODO: Transpose?

	for idx := 0; idx < l.output.Len(); idx++ {
		l.output.SetVec(idx, l.activation(l.output.AtVec(idx)))
	}

	return l.output
}

func (l *Layer) updateWeights(inputs *mat.VecDense, learningRate float64) {
	alpha := learningRate

	// Compute: Weights = alpha * Input^T * Delta + 1 * Weights
	l.scratch.Outer(alpha, l.delta, inputs)
	l.weights.Add(l.weights, l.scratch)
}

type Network struct {
	layers       []Layer
}

/* Unbiased new network */
func NewNetwork(layerSizes []int) *Network {
	layers := []Layer{}

	for idx, numInputs := range layerSizes[:len(layerSizes)-1] {
		numOutputs := layerSizes[idx+1]
		layer := NewLayer(numInputs, numOutputs)
		layers = append(layers, layer)
	}

	return &Network{
		layers: layers,
	}
}

func (n *Network) snapshot(prefix string) {
	for idx, layer := range n.layers {
		layer.snapshot(fmt.Sprintf(`%s-%d.layer`, prefix, idx))
	}
}

func (n *Network) restore(prefix string) {
	for idx, layer := range n.layers {
		layer.restore(fmt.Sprintf(`%s-%d.layer`, prefix, idx))
	}
}

func (n *Network) forward(inputs []float64) []float64 {
	output := mat.NewVecDense(len(inputs), inputs)

	for _, layer := range n.layers {
		output = layer.forward(output)
	}

	res := []float64{}
	for idx := 0; idx < output.Len(); idx++ {
		res = append(res, output.AtVec(idx))
	}
	return res
}

func (n *Network) backprop(inputs []float64, error []float64, learningRate float64) {
	localError := mat.NewVecDense(len(error), error)
	for layer_idx := len(n.layers) - 1; layer_idx >= 0; layer_idx-- {
		localError = n.layers[layer_idx].computeGradient(localError)
	}

	localInput := mat.NewVecDense(len(inputs), inputs)
	for _, layer := range n.layers {
		layer.updateWeights(localInput, learningRate)
		localInput = layer.output
	}
}

func (n *Network) error(outputs, targets []float64) []float64 {
	error := []float64{}

	for idx, t := range targets {
		error = append(error, t-outputs[idx])
	}

	return error
}

type Sample struct {
	input  []float64
	target []float64
}
