package network

import (
	"archive/tar"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strconv"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"

	"github.com/farhaven/nn-go/activation"
)

type layer struct {
	weights    *mat.Dense
	delta      *mat.VecDense
	output     *mat.VecDense
	scratch    *mat.Dense // Scratch buffer for weight updates
	activation activation.Activation
}

func newLayer(inputs, outputs int, activation activation.Activation) layer {
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

func (l *layer) clone() *layer {
	clone := layer{
		activation: l.activation,
		weights:    mat.DenseCopyOf(l.weights),
		delta:      mat.VecDenseCopyOf(l.delta),
		output:     mat.VecDenseCopyOf(l.output),
		scratch:    mat.DenseCopyOf(l.scratch),
	}

	return &clone
}

func (l *layer) WriteTo(w io.Writer) (int64, error) {
	sz, err := l.weights.MarshalBinaryTo(w)
	return int64(sz), err
}

var _ io.WriterTo = &layer{}

func (l *layer) encodedSize() (int, error) {
	buf, err := l.weights.MarshalBinary()
	if err != nil {
		return 0, err
	}

	return len(buf), nil
}

func (l *layer) ReadFrom(r io.Reader) (int64, error) {
	var weights mat.Dense

	sz, err := weights.UnmarshalBinaryFrom(r)
	if err != nil {
		return 0, err
	}

	l.weights = &weights

	return int64(sz), nil
}

var _ io.ReaderFrom = &layer{}

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
	for idx, val := range inputs.RawVector().Data {
		if math.IsNaN(val) {
			panic(fmt.Sprintf("NaN layer input at %v", idx))
		}
	}

	l.output.MulVec(l.weights, inputs)

	for idx := 0; idx < l.output.Len(); idx++ {
		f := l.activation.Forward(l.output.AtVec(idx))
		if math.IsNaN(f) {
			panic(fmt.Sprintf("NaN layer output, was %v before activation", l.output.AtVec(idx)))
		}

		l.output.SetVec(idx, f)
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

// LayerConf represents a configuration for one single layer in the network
type LayerConf struct {
	Inputs     int
	Activation activation.Activation
}

// NewNetwork creates a new neural network with the desired layer configurations.
// The activation is ignored for the first layer and has to be set to nil.
//
// The following creates a fully connected 2x3x1 network with sigmoid activation between all layers:
//
//  config := []LayerConf{
//    LayerConf{Inputs: 2, Activation: nil},
//    LayerConf{Inputs: 3, Activation: SigmoidActivation{}},
//    LayerConf{Inputs: 1, Activation: SigmoidActivation{}},
//  }
//  net := network.NewNetwork(config)
func New(layerConfigs []LayerConf) (*Network, error) {
	if layerConfigs[0].Activation != nil {
		return nil, errors.New(`First activation has to be nil!`)
	}

	layers := []*layer{}

	for idx, conf := range layerConfigs[1:] {
		numOutputs := conf.Inputs
		numInputs := layerConfigs[idx].Inputs

		layer := newLayer(numInputs, numOutputs, conf.Activation)
		layers = append(layers, &layer)
	}

	return &Network{
		layers: layers,
	}, nil
}

func (n *Network) Clone() *Network {
	clone := Network{}

	for _, l := range n.layers {
		clone.layers = append(clone.layers, l.clone())
	}

	return &clone
}

type writeCounter struct {
	w io.Writer
	c int64
}

func (w *writeCounter) Write(data []byte) (int, error) {
	sz, err := w.w.Write(data)
	w.c += int64(sz)
	return sz, err
}

// WriteTo writes a snapshot of n to w.
func (n *Network) WriteTo(w io.Writer) (int64, error) {
	wc := writeCounter{w: w}
	tw := tar.NewWriter(&wc)

	for idx, layer := range n.layers {
		sz, err := layer.encodedSize()
		if err != nil {
			return wc.c, fmt.Errorf("getting layer size for %d: %w", idx, err)
		}

		err = tw.WriteHeader(&tar.Header{
			Name: "layer-" + strconv.Itoa(idx),
			Size: int64(sz),
		})
		if err != nil {
			return wc.c, fmt.Errorf("creating entry for layer %d: %w", idx, err)
		}

		_, err = layer.WriteTo(tw)
		if err != nil {
			return wc.c, fmt.Errorf("persisting layer %d: %w", idx, err)
		}

		err = tw.Flush()
		if err != nil {
			return wc.c, fmt.Errorf("flushing layer %d: %w", idx, err)
		}
	}

	err := tw.Close()
	if err != nil {
		return wc.c, err
	}

	return wc.c, nil
}

var _ io.WriterTo = &Network{}

type readCounter struct {
	r io.Reader
	c int64
}

func (r *readCounter) Read(data []byte) (int, error) {
	sz, err := r.r.Read(data)
	r.c += int64(sz)
	return sz, err
}

// ReadFrom restores a network that was previously saved with `Snapshot`.
//
// The result is undefined if the network architecture differs. You will likely get panics or weird errors
// when using or training a network that was restored from different parameters.
//
// TODO: Persist network architecture and validate on restore.
func (n *Network) ReadFrom(r io.Reader) (int64, error) {
	rc := readCounter{r: r}
	tr := tar.NewReader(&rc)

	for idx, layer := range n.layers {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return rc.c, fmt.Errorf("skipping archive header: %w", err)
		}

		if hdr.Name != "layer-"+strconv.Itoa(idx) {
			return rc.c, fmt.Errorf("unexpected archive entry %q, expected one for layer %d", hdr.Name, idx)
		}

		_, err = layer.ReadFrom(tr)
		if err != nil {
			return rc.c, fmt.Errorf("restoring layer %d: %w", idx, err)
		}
	}

	return rc.c, nil
}

var _ io.ReaderFrom = &Network{}

// Forward performs a forward pass through the network for the given inputs.
// The returned value is the output of the uppermost layer of neurons.
func (n *Network) Forward(inputs []float64) []float64 {
	output := mat.NewVecDense(len(inputs), inputs)

	for layerIdx, layer := range n.layers {
		for outIdx, o := range output.RawVector().Data {
			if math.IsNaN(o) {
				panic(fmt.Sprintf("NaN layer output at %d (layer %d)", outIdx, layerIdx))
			}
		}

		output = layer.forward(output)
	}

	res := []float64{}
	for idx := 0; idx < output.Len(); idx++ {
		if math.IsNaN(output.AtVec(idx)) {
			panic("NaN output value")
		}
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
//  error := Error(output, target)
//  net.Backprop(input, error, 0.1) // Perform back propagation with learning rate 0.1
func (n *Network) Backprop(inputs, error []float64, learningRate float64) {
	localError := mat.NewVecDense(len(error), error)
	for idx := len(n.layers) - 1; idx >= 0; idx-- {
		localError = n.layers[idx].computeGradient(localError)
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
func Error(outputs, targets []float64) []float64 {
	error := make([]float64, len(targets))

	for idx, t := range targets {
		error[idx] = t - outputs[idx]
	}

	return error
}
