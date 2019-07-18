package main

import (
	"log"
	"math"
	"math/rand"
	"os"

	network "github.com/farhaven/nn-go"
)

const numEpochs = 300

func maxIdx(values []float64) int {
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

func indexProducer(maxIdx int, c chan int) {
	for {
		for _, idx := range rand.Perm(maxIdx) {
			c <- idx
		}
	}
}

func trainNetwork(net *network.Network, samples []mnistSample) {
	logger := log.New(os.Stdout, `[TRAIN] `, log.LstdFlags)
	logger.Println(`attempting to load network layers from snapshot`)

	net.Restore(`mnist-network`)

	targetMSE := 0.0005
	learningRate := float64(0.01)

	valSize := int(float64(len(samples)) * 0.1) // keep 10% as validation samples
	validationSamples := samples[:valSize]
	trainingSamples := samples[valSize:]

	indexChan := make(chan int)
	go indexProducer(len(trainingSamples), indexChan)

	for epoch := 0; epoch < numEpochs; epoch++ {
		meanMSE := float64(0)

		// Randomize samples
		for batch := 0; batch < len(trainingSamples); batch++ {
			s := trainingSamples[<-indexChan]
			output := net.Forward(s.input)
			error := net.Error(output, s.target)
			net.Backprop(s.input, error, learningRate)

			mse := float64(0)
			for _, e := range error {
				mse += math.Pow(e, 2)
			}
			if math.IsNaN(mse) {
				panic(`NaN mse. Error too high? Check bounds of activation!`)
			}
			meanMSE += mse / float64(len(error) + 1)
		}

		meanMSE /= float64(len(samples) + 1)

		errors := 0
		for _, s := range validationSamples {
			output := net.Forward(s.input)
			label := maxIdx(output)
			if label != maxIdx(s.target) {
				errors += 1
			}
		}
		errorRate := float64(errors) / float64(len(validationSamples)+1)

		logger.Printf(`epoch % 3d: %d/%d -> %.3f%% error, mse: %.5f`, epoch, errors, len(validationSamples), errorRate * 100, meanMSE)

		if epoch % 10 == 0 {
			learningRate = learningRate * 0.9
			logger.Println(`adjusted learning rate to`, learningRate)
		}

		// Make a snapshot of the network after each epoch
		net.Snapshot(`mnist-network`)

		if meanMSE <= targetMSE {
			logger.Println(`target mse reached after`, epoch, `training epochs`)
			break
		}
	}
}
