package network

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	weights    *mat.Dense
	delta      *mat.VecDense
	output     *mat.VecDense
	scratch    *mat.Dense // Scratch buffer for weight updates
	activation Activation
}

func NewLayer(inputs, outputs int, activation Activation) Layer {
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
		activation: activation,
	}
}

func (l *Layer) snapshot(path string) error {
	fh, err := os.Create(path)
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf(`while creating snapshot %s`, path))
	}
	defer fh.Close()

	_, err = l.weights.MarshalBinaryTo(fh)
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf(`while unmarshaling snapshot %s`, path))
	}

	return nil
}

func (l *Layer) restore(path string) error {
	fh, err := os.Open(path)
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf(`while reading snapshot %s`, path))
	}
	defer fh.Close()

	var weights mat.Dense
	_, err = weights.UnmarshalBinaryFrom(fh)
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf(`while unmarshaling snapshot %s`, path))
	}
	l.weights = &weights

	return nil
}

func (l *Layer) computeGradient(error *mat.VecDense) *mat.VecDense {
	for idx := 0; idx < error.Len(); idx++ {
		// TODO: See if this can be unified
		e := error.AtVec(idx)
		l.delta.SetVec(idx, e*l.activation.Backward(l.output.AtVec(idx)))
	}

	var res mat.Dense
	res.Mul(mat.Matrix(l.delta).T(), l.weights)

	_, c := l.weights.Dims()

	resVec := mat.NewVecDense(c, nil)
	resVec.CopyVec(res.RowView(0))

	return resVec
}

func (l *Layer) forward(inputs *mat.VecDense) *mat.VecDense {
	l.output.MulVec(l.weights, inputs)

	for idx := 0; idx < l.output.Len(); idx++ {
		l.output.SetVec(idx, l.activation.Forward(l.output.AtVec(idx)))
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
	layers []*Layer
}

/* Unbiased new network */
func NewNetwork(layerSizes []int, activation Activation) *Network {
	layers := []*Layer{}

	for idx, numInputs := range layerSizes[:len(layerSizes)-1] {
		numOutputs := layerSizes[idx+1]
		layer := NewLayer(numInputs, numOutputs, activation)
		layers = append(layers, &layer)
	}

	return &Network{
		layers: layers,
	}
}

func (n *Network) Snapshot(prefix string) error {
	for idx, layer := range n.layers {
		if err := layer.snapshot(fmt.Sprintf(`%s-%d.layer`, prefix, idx)); err != nil {
			return err
		}
	}

	return nil
}

func (n *Network) Restore(prefix string) error {
	for idx, layer := range n.layers {
		if err := layer.restore(fmt.Sprintf(`%s-%d.layer`, prefix, idx)); err != nil {
			return err
		}
	}

	return nil
}

func (n *Network) Forward(inputs []float64) []float64 {
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

func (n *Network) Backprop(inputs []float64, error []float64, learningRate float64) {
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

func (n *Network) Error(outputs, targets []float64) []float64 {
	error := []float64{}

	for idx, t := range targets {
		error = append(error, t-outputs[idx])
	}

	return error
}
