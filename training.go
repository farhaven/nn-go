package main

import (
	"log"
	"math"
	"os"
)

const numEpochs = 300

func MaxIdx(values []float64) int {
	maxSeen := math.Inf(-1)
	maxIdx := 0

	for idx, val := range values {
		if val > maxSeen {
			maxSeen = val
			maxIdx = idx
		}
	}

	return maxIdx
}

func trainNetwork(net *Network, samples []Sample) {
	logger := log.New(os.Stdout, `[TRAIN] `, log.LstdFlags)

	targetMSE := 0.005
	learningRate := float64(0.1)

	valSize := int(float64(len(samples)) * 0.1) // keep 10% as validation samples
	validationSamples := samples[:valSize]
	trainingSamples := samples[valSize:]

	for epoch := 0; epoch < numEpochs; epoch++ {
		meanSquaredError := float64(0)

		for _, s := range trainingSamples {
			output := net.forward(s.input)
			error := net.error(output, s.target)
			net.backprop(s.input, error, learningRate)

			for _, e := range error {
				meanSquaredError += math.Pow(e, 2)
			}
		}

		meanSquaredError /= float64(len(samples) + 1) * float64(len(samples[0].target))

		errors := 0
		for _, s := range validationSamples {
			output := net.forward(s.input)
			label := MaxIdx(output)
			if label != MaxIdx(s.target) {
				errors += 1
			}
		}

		logger.Println(epoch, errors, `errors out of`, len(validationSamples), `tests ->`, float64(errors)/float64(len(validationSamples) + 1), `error rate`, `mse`, meanSquaredError)

		if meanSquaredError <= targetMSE {
			logger.Println(`target mse reached after`, epoch, `training epochs`)
		}
	}
}
