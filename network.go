package network

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

type layer struct {
	weights    *mat.Dense
	delta      *mat.VecDense
	output     *mat.VecDense
	scratch    *mat.Dense // Scratch buffer for weight updates
	activation Activation
}

func newLayer(inputs, outputs int, activation Activation) layer {
	// Initialize layer with random weights
	weights := mat.NewDense(outputs, inputs, nil)
	weights.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64()
	}, weights)

	return layer{
		weights:    weights,
		delta:      mat.NewVecDense(outputs, nil),
		output:     mat.NewVecDense(outputs, nil),
		scratch:    mat.NewDense(outputs, inputs, nil),
		activation: activation,
	}
}

func (l *layer) snapshot(path string) error {
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

func (l *layer) restore(path string) error {
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

func (l *layer) computeGradient(error *mat.VecDense) *mat.VecDense {
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

func (l *layer) forward(inputs *mat.VecDense) *mat.VecDense {
	l.output.MulVec(l.weights, inputs)

	for idx := 0; idx < l.output.Len(); idx++ {
		l.output.SetVec(idx, l.activation.Forward(l.output.AtVec(idx)))
	}

	return l.output
}

func (l *layer) updateWeights(inputs *mat.VecDense, learningRate float64) {
	alpha := learningRate

	// Compute: Weights = alpha * Input^T * Delta + 1 * Weights
	l.scratch.Outer(alpha, l.delta, inputs)
	l.weights.Add(l.weights, l.scratch)
}

// Network is structure that represents an unbiased neural network
type Network struct {
	layers []*layer
}

// LayerConfiguration represents a configuration for one single layer in the network
type LayerConfiguration struct {
	NumNodes   int
	Activation Activation
}

// NewNetwork creates a new neural network with the desired layer configurations.
// The activation is ignored for the first layer and has to be set to nil.
//
// The following creates a fully connected 2x3x1 network with sigmoid activation between all layers:
//
//  config := []LayerConfiguration{
//    LayerConfiguration{2, nil},
//    LayerConfiguration{3, SigmoidActivation{}},
//    LayerConfiguration{1, SigmoidActivation{}},
//  }
//  net := network.NewNetwork(config)
func NewNetwork(layerConfigs []LayerConfiguration) (*Network, error) {
	if layerConfigs[0].Activation != nil {
		return nil, errors.New(`First activation has to be nil!`)
	}

	layers := []*layer{}

	for idx, conf := range layerConfigs[1:len(layerConfigs)] {
		numOutputs := conf.NumNodes
		activation := conf.Activation
		numInputs := layerConfigs[idx].NumNodes

		layer := newLayer(numInputs, numOutputs, activation)
		layers = append(layers, &layer)
	}

	return &Network{
		layers: layers,
	}, nil
}

// Snapshot stores a snapshot of all layers to files prefixed with `prefix`.
// The files are suffixed with the layer number and the string `.layer`.
func (n *Network) Snapshot(prefix string) error {
	for idx, layer := range n.layers {
		if err := layer.snapshot(fmt.Sprintf(`%s-%d.layer`, prefix, idx)); err != nil {
			return err
		}
	}

	return nil
}

// Restore restores a network that was previously saved with `Snapshot`.
//
// The result is undefined if the network architecture differs.
func (n *Network) Restore(prefix string) error {
	for idx, layer := range n.layers {
		if err := layer.restore(fmt.Sprintf(`%s-%d.layer`, prefix, idx)); err != nil {
			return err
		}
	}

	return nil
}

// Forward performs a forward pass through the network for the given inputs.
// The returned value is the output of the uppermost layer of neurons.
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

// Backprop performs one pass of back propagation through the network for the given input, error and learning rate.
//
// Before Backprop is called, you need to do one forward pass for the input with Forward. A typical usage
// looks like this:
//
//  input := []float64{0, 1.0, 2.0}
//  target := []float64{0, 1}
//  output := net.Forward(input)
//  error := net.Error(output, target)
//  net.Backprop(input, error, 0.1) // Perform back propagation with learning rate 0.1
func (n *Network) Backprop(inputs, error []float64, learningRate float64) {
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

// Error computes the error of the given outputs when compared to the given targets.
//
// This is intended to be used during training. See the documentation for Backprop for an example usage.
func (n *Network) Error(outputs, targets []float64) []float64 {
	error := []float64{}

	for idx, t := range targets {
		error = append(error, t-outputs[idx])
	}

	return error
}
