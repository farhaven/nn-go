/*
Package network is a simple implementation of a nonbiased neural network.

The networks created by this package can be trained with backpropagation and use a variety of activation
functions.

For example, the following code trains a simple 2x3x1 neural network the XOR function:

	config := []network.LayerConfiguration{
		network.LayerConfiguration{2, nil},
		network.LayerConfiguration{3, network.LeakyRELUActivation{Leak: 0.01}},
		network.LayerConfiguration{1, network.LeakyRELUActivation{Leak: 0.01}},
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
*/
package network
