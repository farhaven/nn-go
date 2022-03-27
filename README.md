# network
Package network is a simple implementation of a nonbiased neural network.

The networks created by this package can be trained with backpropagation and use
a variety of activation functions.

For example, the following code trains a simple 2x3x1 neural network the XOR
function:

    config := []network.LayerConf{
    	network.LayerConf{Inputs: 2},
    	network.LayerConf{Inputs:3, Activation: activation.LeakyReLU{Leak: 0.01}},
    	network.LayerConf{Inputs:1, Activation: activation.LeakyReLU{Leak: 0.01}},
    }
    net, err := network.NewNetwork(config)
    if err != nil {
    	log.Fatalln(`can't create network`, err)
    }

    // Training samples
    samples := map[[2]float64][]float64{
    	[2]float64{0, 0}: []float64{0},
    	[2]float64{0, 1}: []float64{1},
    	[2]float64{1, 0}: []float64{1},
    	[2]float64{1, 1}: []float64{0},
    }

    targetMSE := 0.005  // Desired Mean Squared Error
    learningRate := 0.1 // Learning rate for the network, larger is faster, smaller is more accurate

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
    }

    log.Println(`Took`, iter, `iterations to reach target MSE`, targetMSE)

    for input, target := range samples {
    	log.Println(`Input:`, input, `Target:`, target, `Output:`, net.Forward(input[:])
    }

## Usage

#### func  Error

```go
func Error(outputs, targets []float64) []float64
```
Error computes the error of the given outputs when compared to the given
targets.

This is intended to be used during training. See the documentation for Backprop
for an example usage.

#### type LayerConf

```go
type LayerConf struct {
	Inputs     int
	Activation activation.Activation
}
```

LayerConf represents a configuration for one single layer in the network

#### type Network

```go
type Network struct {
}
```

Network is structure that represents an unbiased neural network

#### func  NewNetwork

```go
func NewNetwork(layerConfigs []LayerConf) (*Network, error)
```
NewNetwork creates a new neural network with the desired layer configurations.
The activation is ignored for the first layer and has to be set to nil.

The following creates a fully connected 2x3x1 network with sigmoid activation
between all layers:

    config := []LayerConf{
      LayerConf{Inputs: 2, Activation: nil},
      LayerConf{Inputs: 3, Activation: SigmoidActivation{}},
      LayerConf{Inputs: 1, Activation: SigmoidActivation{}},
    }
    net := network.NewNetwork(config)

#### func (*Network) Backprop

```go
func (n *Network) Backprop(inputs, error []float64, learningRate float64)
```
Backprop performs one pass of back propagation through the network for the given
input, error and learning rate.

Before Backprop is called, you need to do one forward pass for the input with
Forward. A typical usage looks like this:

    input := []float64{0, 1.0, 2.0}
    target := []float64{0, 1}
    output := net.Forward(input)
    error := Error(output, target)
    net.Backprop(input, error, 0.1) // Perform back propagation with learning rate 0.1

#### func (*Network) Clone

```go
func (n *Network) Clone() *Network
```

#### func (*Network) Forward

```go
func (n *Network) Forward(inputs []float64) []float64
```
Forward performs a forward pass through the network for the given inputs. The
returned value is the output of the uppermost layer of neurons.

#### func (*Network) Restore

```go
func (n *Network) Restore(prefix string) error
```
Restore restores a network that was previously saved with `Snapshot`.

The result is undefined if the network architecture differs.

#### func (*Network) Snapshot

```go
func (n *Network) Snapshot(prefix string) error
```
Snapshot stores a snapshot of all layers to files prefixed with `prefix`. The
files are suffixed with the layer number and the string `.layer`.
