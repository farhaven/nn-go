package main

import (
	"log"
	"os"
	"sort"
	"sync"
)

const numParallel = 22
const numEpochs = 10000
const epochSlice = 75 // Number of survivors for each epoch

func trainNetworks(networks []*Network, samples []Sample) []*Network {
	logger := log.New(os.Stdout, `[TRAIN] `, log.LstdFlags)

	valSize := int(float64(len(samples)) * 0.1) /* keep 10% as validation samples */
	validationSamples := samples[:valSize]
	trainingSamples := samples[valSize:]

	workerChan := make(chan *Network)
	workerWg := sync.WaitGroup{}

	for idx := 0; idx < numParallel; idx++ {
		go func(idx int) {
			for net := range workerChan {
				net.updateTotalError(trainingSamples)
				workerWg.Done()
			}
		}(idx)
	}

trainingLoop:
	for epoch := 0; epoch < numEpochs; epoch++ {
		for _, net := range networks {
			workerWg.Add(1)
			workerChan <- net
		}
		workerWg.Wait()

		sort.Slice(networks, func(i, j int) bool {
			return networks[i].performance() > networks[j].performance()
		})

		/* Cull non-survivors */
		networks = networks[:epochSlice]

		for len(networks) <= epochSlice {
			/* Clone each survivor and mutate the clone */
			newNetworks := []*Network{}
			for _, net := range networks {
				clone := net.Clone()
				clone.mutate(10)
				net.mutate(1)
				newNetworks = append(newNetworks, clone)
			}
			networks = append(networks, newNetworks...)

			bestPerf := networks[0].performance()
			bestError := networks[0].averageError
			logger.Println(`epoch`, epoch, `best performance`, bestPerf, `best error`, bestError)
			if bestPerf == 1 && epoch > 0 {
				break trainingLoop
			}
			if epoch%10 == 0 {
				errors := 0
				for _, s := range validationSamples {
					networks[0].feed(s.inputs)
					output := MaxIdx(networks[0].getOutput())
					target := MaxIdx(s.targets)
					if output != target {
						errors += 1
					}
				}
				logger.Println("\t", errors, `errors out of`, len(validationSamples), `tests ->`, float64(errors)/float64(len(validationSamples)), `error rate`)
			}

			/* Cull duplicates */
			structures := make(map[string]*Network)
			for _, net := range networks {
				structures[net.structuralHash()] = net
			}
			networks = []*Network{}
			for _, net := range structures {
				networks = append(networks, net)
			}
		}

		for _, net := range networks {
			net.dedupEdges()
		}
	}

	for _, net := range networks {
		net.updateTotalError(samples)
		net.removeDeadEnds()
	}

	sort.Slice(networks, func(i, j int) bool {
		return networks[i].performance() > networks[j].performance()
	})

	close(workerChan)

	return networks
}
