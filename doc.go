/*
Package network is a simple implementation of a nonbiased neural network.

The networks created by this package can be trained with backpropagation and use a variety of activation
functions.

For example, the following code trains a simple 2x3x1 neural network the XOR function:

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
*/
package network
