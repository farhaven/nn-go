package main

import (
	"log"
	"math"
	"os"
)

const numEpochs = 10

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

func trainNetwork(network *Network, samples []Sample) {
	logger := log.New(os.Stdout, `[TRAIN] `, log.LstdFlags)

	valSize := int(float64(len(samples)) * 0.1) /* keep 10% as validation samples */
	validationSamples := samples[:valSize]
	trainingSamples := samples[valSize:]

	for epoch := 0; epoch < numEpochs; epoch++ {
		logger.Println(`starting epoch`, epoch)

		network.train(trainingSamples)

		network.updateAverageError(validationSamples)

		logger.Println(`epoch`, epoch, `done. average error: `, network.averageError)

		if epoch%10 == 0 {
			errors := 0
			for _, s := range validationSamples {
				output := MaxIdx(network.feed(s.inputs, nil))
				target := MaxIdx(s.targets)
				if output != target {
					errors += 1
				}
			}
			logger.Println("\t", errors, `errors out of`, len(validationSamples), `tests ->`, float64(errors)/float64(len(validationSamples)), `error rate`)
		}
	}
}
