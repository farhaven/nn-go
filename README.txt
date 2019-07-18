package network // import "github.com/farhaven/nn-go"

Package network is a simple implementation of a nonbiased neural network.

The networks created by this package can be trained with backpropagation and
use a variety of activation functions.

For example, the following code trains a simple 2x3x1 neural network the XOR
function:

    type Sample [2][]float64  // Input and output
    samples := []Sample{
    	Sample{[]float64{0, 0}, []float64{0}},
    	Sample{[]float64{0, 1}, []float64{1}},
    	Sample{[]float64{1, 0}, []float64{1}},
    	Sample{[]float64{1, 1}, []float64{0}},
    }

    learningRate := 0.75
    targetMSE := 0.005  // Target mean squared error

    net := network.NewNetwork([]int{2, 3, 1}, network.LeakyRELUActivation{Leak: 0.01})

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

    	log.Println(`Iter`, iter, `MSE`, meanSquaredError)
    }

    // Print out network output for each sample:
    for i, s := range samples {
    	log.Printf(`Sample %d: %v -> %v`, i, s, net.Forward(s[0]))
    }

TYPES

type Activation interface {
	Forward(float64) float64
	Backward(float64) float64
}
    Activation represents an activation function.

type ELUActivation struct {
	A float64
}
    ELUActivation is an Exponential Linear Unit activation. It computes max(A *
    (e^x - 1), x).

func (e ELUActivation) Backward(x float64) float64
func (e ELUActivation) Forward(x float64) float64
type Layer struct {
	// Has unexported fields.
}

func NewLayer(inputs, outputs int, activation Activation) Layer
type LeakyRELUActivation struct {
	Leak float64
	Cap  float64
}
    LeakyRELUActivation is a Leaky Rectified Linear Unit activation with
    activation cap.

    It computes min(x, Cap) if x > 0, otherwise it computes x * Leak

    If Cap is 0, the unit is uncapped.

func (r LeakyRELUActivation) Backward(x float64) float64
func (r LeakyRELUActivation) Forward(x float64) float64
type Network struct {
	// Has unexported fields.
}
    Network is structure that represents an unbiased neural network

func NewNetwork(layerSizes []int, activation Activation) *Network
    NewNetwork creates a new neural network with the desired layer sizes and
    activation function.

    The following creates a fully connected 2x3x1 network with sigmoid
    activation:

    net := network.NewNetwork([]int{2, 3, 1}, SigmoidActivation{})

func (n *Network) Backprop(inputs, error []float64, learningRate float64)
    Backprop performs one pass of back propagation through the network for the
    given input, error and learning rate.

    Before Backprop is called, you need to do one forward pass for the input
    with Forward. A typical usage looks like this:

    input := []float64{0, 1.0, 2.0}
    target := []float64{0, 1}
    output := net.Forward(input)
    error := net.Error(output, target)
    net.Backprop(input, error, 0.1) // Perform back propagation with learning rate 0.1

func (n *Network) Error(outputs, targets []float64) []float64
    Error computes the error of the given outputs when compared to the given
    targets.

    This is intended to be used during training. See the documentation for
    Backprop for an example usage.

func (n *Network) Forward(inputs []float64) []float64
    Forward performs a forward pass through the network for the given inputs.
    The returned value is the output of the uppermost layer of neurons.

func (n *Network) Restore(prefix string) error
    Restore restores a network that was previously saved with `Snapshot`.

    The result is undefined if the network architecture differs.

func (n *Network) Snapshot(prefix string) error
    Snapshot stores a snapshot of all layers to files prefixed with `prefix`.
    The files are suffixed with the layer number and the string `.layer`.

type SigmoidActivation struct{}
    SigmoidActivation is a sigmoid activation function. It computes 1/(1 +
    e^(-x)).

func (s SigmoidActivation) Backward(x float64) float64
func (s SigmoidActivation) Forward(x float64) float64
type TanhActivation struct{}
    TanhActivation computes tanh(x) as the activation function.

func (t TanhActivation) Backward(x float64) float64
func (t TanhActivation) Forward(x float64) float64
